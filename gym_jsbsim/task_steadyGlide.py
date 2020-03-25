#TODO:: check if all imports are really needed
import gym
import numpy as np
import random
import types
import math
import enum
import warnings
from collections import namedtuple
import gym_jsbsim.properties as prp
from gym_jsbsim import assessors, rewards, utils
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.properties import BoundedProperty, Property
from gym_jsbsim.aircraft import Aircraft
from gym_jsbsim.rewards import RewardStub
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Tuple, NamedTuple, Type


from gym_jsbsim.tasks import FlightTask, Shaping


class RandomParams(namedtuple('RandomParams', ['mean', 'std_dev'])):
    ...

class SteadyRollGlideTask(FlightTask):
    """
    A task in which the agent shall maintain a
    - steady glide angle (adjustable)
    - steady banking angle (adjustable)
    """
    INTEGRAL_DECAY = 1   #just a starting value. TODO: make configurable if it works out
    TARGET_GLIDE_ANGLE_DEG_default = -6   #TODO: this is arbitrary
    TARGET_ROLL_ANGLE_DEG_default  = -10
    INITIAL_ALTITUDE_FT_default = 6000
    EPISODE_TIME_S_default = 60.  #TODO: make configurable
    # The scaling factor specifies at what error from the target the agent receives 0.5 reward or potential out of a maximum of 1.0.
    GLIDE_ANGLE_DEG_ERROR_SCALING = 1
    ROLL_ANGLE_DEG_ERROR_SCALING = 1
    MIN_STATE_QUALITY = 0.0  # terminate if state 'quality' is less than this
    MAX_GLIDE_ANGLE_DEVIATION_DEG = 45
    MAX_ROLL_ANGLE_DEVIATION_DEG = 90
    #those are additional properties going into the sim[] object
    prop_glide_angle_error_deg = BoundedProperty('error/glideAngle-error-deg',
                                      'error to desired glide angle [deg]', -180, 180)
    prop_glide_angle_error_integral_deg_sec = BoundedProperty('error/glideAngle-error-integral-deg-sec',
                                      'decayed sum of the glide angle error [deg*sec]', -1800, 1800)    #arbitrary boundary, add clipping if required
    prop_roll_angle_error_deg = BoundedProperty('error/rollAngle-error-deg',
                                        'error to desired roll/banking angle [deg]', -180, 180)
    prop_roll_angle_error_integral_deg_sec = BoundedProperty('error/rollAngle-error-integral-deg-sec',
                                        'decayed sum of the roll/banking angle error [deg*sec]', -1800, 1800)   #arbitrary boundary, add clipping if required
    prp_delta_cmd_elevator = BoundedProperty('info/delta-cmd-elevator', 
                                'the actuation travel for the elevator command since the last step',
                                prp.elevator_cmd.min - prp.elevator_cmd.max, prp.elevator_cmd.max - prp.elevator_cmd.min)
    prp_delta_cmd_aileron = BoundedProperty('info/delta-cmd-aileron', 
                                'the actuation travel for the aileron command since the last step',
                                prp.aileron_cmd.min - prp.aileron_cmd.max, prp.aileron_cmd.max - prp.aileron_cmd.min)

    action_variables = (prp.elevator_cmd, prp.aileron_cmd)

    def __init__(self, shaping_type: Shaping, step_frequency_hz: float, aircraft: Aircraft,
                 episode_time_s: float = EPISODE_TIME_S_default, positive_rewards: bool = True):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.max_time_s = episode_time_s
        self.step_frequency_hz = step_frequency_hz
        self.aircraft = aircraft
        self.extra_state_variables = (  self.prop_glide_angle_error_deg #13
                                      , self.prop_roll_angle_error_deg  #14
                                      , prp.steps_left                  #15
                                      , prp.flight_path_deg             #16
                                      , prp.roll_deg                    #17
                                      , prp.indicated_airspeed          #18
                                      , prp.true_airspeed               #19
                                      , prp.setpoint_flight_path_deg    #20
                                      , prp.setpoint_roll_angle_deg     #21
                                      , prp.angleOfAttack_deg           #22
                                      , prp.elevator_cmd                #23 include cmd value into the state to calculate Δ cmd to avoid thrashing 
                                      , prp.aileron_cmd                 #24 include cmd value into the state to calculate Δ cmd to avoid thrashing 
                                      , self.prp_delta_cmd_elevator     #25
                                      , self.prp_delta_cmd_aileron      #26
                                      , prp.pdot_rad_sec2               #27
                                      , self.prop_glide_angle_error_integral_deg_sec     #28
                                      , self.prop_roll_angle_error_integral_deg_sec      #29
                                      )
        self.state_variables = FlightTask.base_state_variables + self.extra_state_variables
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)

        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.setpoints: Dict[Property, float] = {
                  prp.episode_steps: episode_steps
                , prp.setpoint_flight_path_deg: self.TARGET_GLIDE_ANGLE_DEG_default
                , prp.setpoint_roll_angle_deg:  self.TARGET_ROLL_ANGLE_DEG_default
            }
        self.inital_attitude: Dict[Property, float] = {
                  prp.initial_u_fps: self.aircraft.get_cruise_speed_fps()*0.8    #forward speed
                , prp.initial_altitude_ft: self.INITIAL_ALTITUDE_FT_default
                , prp.initial_flight_path_deg: 0
                , prp.initial_roll_deg:0
                , prp.initial_aoa_deg: 1.0    #just an arbitrary value for a reasonable AoA
        }

        super().__init__(assessor, debug = True)

    def make_assessor(self, shaping: Shaping) -> assessors.AssessorImpl:
        base_components = self._make_base_reward_components()
        shaping_components = ()
        return self._select_assessor(base_components, shaping_components, shaping)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        AILERON_CMD_TRAVEL_SCALING_FACTOR  = 2
        ELEVATOR_CMD_TRAVEL_SCALING_FACTOR = 2
        GLIDE_ANGLE_INT_DEG_ERROR_SCALING = self.GLIDE_ANGLE_DEG_ERROR_SCALING / 4
        ROLL_ANGLE_INTEGRAL_DEG_ERROR_SCALING = self.ROLL_ANGLE_DEG_ERROR_SCALING/4
        base_components = (
            rewards.AsymptoticErrorComponent(name='rwd_rollAngle_error',
                                    prop=self.prop_roll_angle_error_deg,
                                    state_variables=self.state_variables,
                                    target=0.0,
                                    potential_difference_based=False,
                                    scaling_factor=self.ROLL_ANGLE_DEG_ERROR_SCALING,
                                    weight=4),
            rewards.AsymptoticErrorComponent(name='rwd_rollAngle_error_Integral',
                                    prop=self.prop_roll_angle_error_integral_deg_sec,
                                    state_variables=self.state_variables,
                                    target=0.0,
                                    potential_difference_based=False,
                                    scaling_factor=ROLL_ANGLE_INTEGRAL_DEG_ERROR_SCALING,
                                    weight=9),
            rewards.LinearErrorComponent(name='rwd_aileron_cmd_travel_error',
                                    prop=self.prp_delta_cmd_aileron,
                                    state_variables=self.state_variables,
                                    target=0.0,
                                    potential_difference_based=False,
                                    scaling_factor=AILERON_CMD_TRAVEL_SCALING_FACTOR,
                                    weight=1),

            rewards.AsymptoticErrorComponent(name='rwd_glideAngle_error',
                        prop=self.prop_glide_angle_error_deg,
                        state_variables=self.state_variables,
                        target=0.0,
                        potential_difference_based=False,
                        scaling_factor=self.GLIDE_ANGLE_DEG_ERROR_SCALING,
                        weight=9),
            rewards.AsymptoticErrorComponent(name='rwd_glideAngle_error_Integral',
                        prop=self.prop_glide_angle_error_integral_deg_sec,
                        state_variables=self.state_variables,
                        target=0.0,
                        potential_difference_based=False,
                        scaling_factor=GLIDE_ANGLE_INT_DEG_ERROR_SCALING,
                        weight=4),
            rewards.LinearErrorComponent(name='rwd_elevator_cmd_travel_error',
                        prop=self.prp_delta_cmd_elevator,
                        state_variables=self.state_variables,
                        target=0.0,
                        potential_difference_based=False,
                        scaling_factor=ELEVATOR_CMD_TRAVEL_SCALING_FACTOR,
                        weight=1),
        )
        return base_components

    def _select_assessor(self, base_components: Tuple[rewards.RewardComponent, ...],
                         shaping_components: Tuple[rewards.RewardComponent, ...],
                         shaping: Shaping) -> assessors.AssessorImpl:
        if shaping is Shaping.STANDARD:
            return assessors.AssessorImpl(base_components, shaping_components,
                                          positive_rewards=self.positive_rewards)
        else:
            #TODO: This may or may not be useful in the future
            potential_based_components = ()
            ...
        if shaping is Shaping.EXTRA:
            return assessors.AssessorImpl(base_components, potential_based_components,
                                          positive_rewards=self.positive_rewards)
        elif shaping is Shaping.EXTRA_SEQUENTIAL:
            #TODO: This may or may not be useful in the future
            glideAngle_error_deg, rollAngle_error_deg = base_components
            # make the wings_level shaping reward dependent on facing the correct direction
            dependency_map = {}
            return assessors.ContinuousSequentialAssessor(base_components, potential_based_components,
                                                          potential_dependency_map=dependency_map,
                                                          positive_rewards=self.positive_rewards)

    def get_initial_conditions(self) -> Dict[Property, float]:
        extra_conditions = {prp.initial_u_fps: self.inital_attitude[prp.initial_u_fps],    #forward speed
                            prp.initial_v_fps: 0,   # side component of speed; shall be 0 in steady flight
                            prp.initial_w_fps: 0,   # down component of speed; shall be 0 in steady flight
                            prp.initial_p_radps: 0, # angular speed roll
                            prp.initial_q_radps: 0,
                            prp.initial_r_radps: 0,
                            # prp.initial_roc_fpm: 0,   #overridden by flight_path
                            prp.initial_heading_deg: 0,
                            prp.initial_flight_path_deg: self.inital_attitude[prp.initial_flight_path_deg],  #to change the initial flightpath angle, change also the sink speed and the pitch
                            prp.initial_roll_deg:        self.inital_attitude[prp.initial_roll_deg],
                            prp.initial_altitude_ft:     self.inital_attitude[prp.initial_altitude_ft],
                            prp.initial_aoa_deg:         self.inital_attitude[prp.initial_aoa_deg],
                            self.prop_glide_angle_error_integral_deg_sec: 0,    #reset the error integral 
                            self.prop_roll_angle_error_integral_deg_sec: 0
                            }
        return {**self.base_initial_conditions, **extra_conditions} #** returns the args as dictionary of named args

    def change_setpoints(self, sim: Simulation, new_setpoints: Dict[Property, float]):
        """
        Changes the setpoint for the task. The changes will take effect within the next environment step. (call to task_step())

        :param sim: The Simulation object of the environment to store the setpoints (needed for visualization)
        :param new_setpoints: A dictionary with new setpoints to be used. New values overwrite old ones.
        """
        if new_setpoints is not None:
            for prop, value in new_setpoints.items():
                sim[prop] = value   #update the setpoints in the simulation model
                self.setpoints[prop] = value    #update the setpoints in the task class
            sim[self.prop_glide_angle_error_integral_deg_sec] = 0 #reset error integrals    TODO: only reset if associated setpoint was changed
            sim[self.prop_roll_angle_error_integral_deg_sec]  = 0 #reset error integrals

    def set_initial_ac_attitude(self, new_initial_conditions: Dict[Property, float]) :#path_angle_gamma_deg, roll_angle_phi_deg, fwd_speed_KAS = None, aoa_deg = 1.0, ):
        """
        Sets the initial flight path angle, the roll angle and the cruise speed to be used as initial conditions
        """
        if new_initial_conditions is not None:
            for prop, value in new_initial_conditions.items():
                self.inital_attitude[prop] = value  #update the setpoints in the task class

    def _update_custom_properties(self, sim: Simulation) -> None:
        #TODO: use some iterable dict with tuples to update all error terms consistently instead of one function per error term
        self._update_glideAngle_error(sim)
        self._update_rollAngle_error(sim)
        self._update_cmd_travel(sim)
        self._decrement_steps_left(sim)

    def _update_glideAngle_error(self, sim: Simulation):
        target_glide_angle_deg = sim[prp.setpoint_flight_path_deg]
        current_glide_angle_deg = sim[prp.flight_path_deg]
        error_deg = utils.reduce_reflex_angle_deg(current_glide_angle_deg - target_glide_angle_deg)
        sim[self.prop_glide_angle_error_deg] = error_deg
        sim[self.prop_glide_angle_error_integral_deg_sec] = sim[self.prop_glide_angle_error_integral_deg_sec] * self.INTEGRAL_DECAY + error_deg * sim[prp.sim_dt]

    def _update_rollAngle_error(self, sim: Simulation):
        target_roll_angle_deg = sim[prp.setpoint_roll_angle_deg]
        current_roll_angle_deg = sim[prp.roll_deg]
        error_deg = utils.reduce_reflex_angle_deg(current_roll_angle_deg - target_roll_angle_deg)
        sim[self.prop_roll_angle_error_deg] = error_deg
        sim[self.prop_roll_angle_error_integral_deg_sec] = sim[self.prop_roll_angle_error_integral_deg_sec] * self.INTEGRAL_DECAY + error_deg * sim[prp.sim_dt]

    
    def _update_cmd_travel(self, sim:Simulation): 
        cmd_elev_idx = self.state_variables.index(prp.elevator_cmd)
        cmd_ail_idx = self.state_variables.index(prp.aileron_cmd)
        if self.last_state:     #skip for the first observation
            delta_elev = sim[prp.elevator_cmd] - self.last_state[cmd_elev_idx]
            delta_ail  = sim[prp.aileron_cmd] - self.last_state[cmd_ail_idx] 
        else:
            delta_elev = sim[prp.elevator_cmd] - 0
            delta_ail  = sim[prp.aileron_cmd] - 0
        sim[self.prp_delta_cmd_elevator] = delta_elev
        sim[self.prp_delta_cmd_aileron] = delta_ail
       
    def _decrement_steps_left(self, sim: Simulation):
        sim[prp.steps_left] -= 1
        #TODO: rather count the steps up and compare with episode_steps; this eases prolonging the episode during runtime

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[prp.steps_left] <= 0
        state_quality = sim[self.last_assessment_reward]
        state_out_of_bounds = state_quality < self.MIN_STATE_QUALITY  # TODO:: issues if sequential?
        return (terminal_step
            or state_out_of_bounds
            or self._glide_or_roll_out_of_bounds(sim)
            or self._altitude_out_of_bounds(sim)
            )  #does snot contribute to return value, but allows me to comment out each condition separately

    def _glide_or_roll_out_of_bounds(self, sim: Simulation) -> bool:
        glideAngle_error_deg = sim[self.prop_glide_angle_error_deg]
        rollAngle_error_deg = sim[self.prop_roll_angle_error_deg]
        return False #TODO: Das muss ich prüfen
        return (abs(glideAngle_error_deg) > self.MAX_GLIDE_ANGLE_DEVIATION_DEG) or \
               (abs(rollAngle_error_deg) > self.MAX_ROLL_ANGLE_DEVIATION_DEG)

    # prevent negative altitudes and terminate before severe crashes happen as this may lead to NaN in state
    # https://github.com/JSBSim-Team/jsbsim/issues/243
    def _altitude_out_of_bounds(self, sim: Simulation) -> bool:
        altitude = sim[prp.altitude_sl_ft]
        if (altitude <= 200):
            print("altitude too small: {}; aborting".format(altitude))
            return True
        else:
            return False

    def _get_out_of_bounds_reward(self, sim: Simulation) -> rewards.Reward:
        """
        if aircraft is out of bounds, we give the largest possible negative reward:
        as if this timestep, and every remaining timestep in the episode was -1.
        """
        reward_scalar = (1 + sim[prp.steps_left]) * -1.
        return RewardStub(reward_scalar, reward_scalar)

    def _reward_terminal_override(self, reward: rewards.Reward, sim: Simulation) -> rewards.Reward:
        if not self.positive_rewards and self._glide_or_roll_out_of_bounds(sim):
            # if using negative rewards, need to give a big negative reward on terminal
            return self._get_out_of_bounds_reward(sim)
        else:
            return reward

    def _new_episode_init(self, sim: Simulation) -> None:
        # entirely override the method of the super class to have the possibility to go with engine off
        # super()._new_episode_init(sim)

        # start with engine off instead of running, so don't start it
        # sim.start_engines()
        # sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)

        sim.raise_landing_gear()
        self._store_reward(RewardStub(1.0, 1.0), sim)   #TODO: What is this? First parameter is not used at all in _store_reward()

        sim[prp.steps_left] = self.setpoints[prp.episode_steps]
        for prop, value in self.setpoints.items():
            sim[prop] = value   #update the setpoints in the simulation model

    def get_props_to_output(self) -> Tuple:
        #TODO: this shall go into a graph or better to go additionally to a graph
        return (prp.u_fps,
                prp.flight_path_deg, prp.setpoint_flight_path_deg, self.prop_glide_angle_error_deg,
                prp.roll_deg, prp.setpoint_roll_angle_deg, self.prop_roll_angle_error_deg,
                self.last_agent_reward, self.last_assessment_reward, prp.steps_left)

    def get_timeline_props_to_output(self) -> Tuple:
        return (prp.flight_path_deg, self.prop_glide_angle_error_deg, prp.elevator,
                prp.roll_deg, self.prop_roll_angle_error_deg, prp.aileron_cmd,)

class SteadyRollAngleTask(SteadyRollGlideTask):

    """
    A task in which the agent shall maintain a
    - steady banking angle (adjustable)

    The only difference to the SteadyRollGlideTask is that the glide path angle does not contribute to the reward.
    The glide path angle error is calculated as well as this is fed into the PID controller for elevator control.
    """

    def make_assessor(self, shaping: Shaping) -> assessors.AssessorImpl:
        base_components = self._make_base_reward_components()
        shaping_components = () #the shaping rewards are ot really what I need; I need reward engineering instead of reward shaping
        return self._select_assessor(base_components, shaping_components, shaping)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        AILERON_CMD_TRAVEL_SCALING_FACTOR = 2  #the max. absolute value of the delta-cmd
        ROLL_ANGLE_INTEGRAL_DEG_ERROR_SCALING = self.ROLL_ANGLE_DEG_ERROR_SCALING/4
        base_components = (
            rewards.AsymptoticErrorComponent(name='rwd_rollAngle_error',
                                    prop=self.prop_roll_angle_error_deg,
                                    state_variables=self.state_variables,
                                    target=0.0,
                                    potential_difference_based=False,
                                    scaling_factor=self.ROLL_ANGLE_DEG_ERROR_SCALING,
                                    weight=4),
            rewards.AsymptoticErrorComponent(name='rwd_rollAngle_error_Integral',
                                    prop=self.prop_roll_angle_error_integral_deg_sec,
                                    state_variables=self.state_variables,
                                    target=0.0,
                                    potential_difference_based=False,
                                    scaling_factor=ROLL_ANGLE_INTEGRAL_DEG_ERROR_SCALING,
                                    weight=9),
            # rewards.QuadraticErrorComponent(name='rwd_aileron_cmd_travel_error',
            rewards.LinearErrorComponent(name='rwd_aileron_cmd_travel_error',
                                    prop=self.prp_delta_cmd_aileron,
                                    state_variables=self.state_variables,
                                    target=0.0,
                                    potential_difference_based=False,
                                    scaling_factor=AILERON_CMD_TRAVEL_SCALING_FACTOR,
                                    weight=1),
        )
        return base_components

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[prp.steps_left] <= 0
        state_quality = sim[self.last_assessment_reward]
        state_out_of_bounds = state_quality < self.MIN_STATE_QUALITY  # TODO:: issues if sequential?
        return (
            terminal_step
            or state_out_of_bounds
            or self._roll_out_of_bounds(sim)
            or self._altitude_out_of_bounds(sim)
        )

    def _roll_out_of_bounds(self, sim: Simulation) -> bool:
        rollAngle_error_deg = sim[self.prop_roll_angle_error_deg]
        return False       #TODO: this shall be evaluated
        return (abs(rollAngle_error_deg) > self.MAX_ROLL_ANGLE_DEVIATION_DEG)
    
    def _select_assessor(self, base_components: Tuple[rewards.RewardComponent, ...],
                         shaping_components: Tuple[rewards.RewardComponent, ...],
                         shaping: Shaping) -> assessors.AssessorImpl:
        if shaping is Shaping.STANDARD:
            # rwd_rollAngle_error, rwd_aileroncmd_travel_error_dep, _ = base_components

            # dependency_map = {rwd_aileroncmd_travel_error_dep: (rwd_rollAngle_error,)} # a Dict(rewards.AsymptoticErrorComponent --> (Tuple))
            # return assessors.ContinuousSequentialAssessor(base_components, shaping_components,
            #                                               base_dependency_map = dependency_map,
            #                                               positive_rewards=self.positive_rewards)

            return assessors.AssessorImpl(base_components, shaping_components,
                                          positive_rewards=self.positive_rewards)
        else:
            small_aileron_commands = rewards.QuadraticErrorComponent(name='small_aileron_commands',
                                                           prop=prp.aileron_cmd,
                                                           state_variables=self.state_variables,
                                                           target=0.0,
                                                           potential_difference_based=True,
                                                           scaling_factor=self.ROLL_ANGLE_DEG_ERROR_SCALING) 
            potential_based_components = (small_aileron_commands,)

        if shaping is Shaping.EXTRA:
            return assessors.AssessorImpl(base_components, potential_based_components,
                                          positive_rewards=self.positive_rewards)
        elif shaping is Shaping.EXTRA_SEQUENTIAL:
            rollAngle_error, _ = base_components
            # make the small_aileron_commands shaping reward dependent on having a small roll error
            dependency_map = {small_aileron_commands: (rollAngle_error,)} # a Dict(rewards.AsymptoticErrorComponent --> (Tuple))
            return assessors.ContinuousSequentialAssessor(base_components, potential_based_components,
                                                          potential_dependency_map=dependency_map,
                                                          positive_rewards=self.positive_rewards)

class SteadyGlideAngleTask(SteadyRollGlideTask):
    """
    A task in which the agent shall maintain a
    - steady glide path angle (adjustable)

    The only difference to the SteadyRollGlideTask is that the roll angle does not contribute to the reward.
    The roll angle error is calculated as well as this is fed into the PID controller for aileron control.
    """

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        ELEVATOR_CMD_TRAVEL_SCALING_FACTOR = 2
        GLIDE_ANGLE_INT_DEG_ERROR_SCALING = self.GLIDE_ANGLE_DEG_ERROR_SCALING / 4
        base_components = (
            rewards.AsymptoticErrorComponent(name='rwd_glideAngle_error',
                        prop=self.prop_glide_angle_error_deg,
                        state_variables=self.state_variables,
                        target=0.0,
                        potential_difference_based=False,
                        scaling_factor=self.GLIDE_ANGLE_DEG_ERROR_SCALING,
                        weight=4),
            rewards.AsymptoticErrorComponent(name='rwd_glideAngle_error_Integral',
                        prop=self.prop_glide_angle_error_integral_deg_sec,
                        state_variables=self.state_variables,
                        target=0.0,
                        potential_difference_based=False,
                        scaling_factor=GLIDE_ANGLE_INT_DEG_ERROR_SCALING,
                        weight=9),
            rewards.LinearErrorComponent(name='rwd_elevator_cmd_travel_error',
                        prop=self.prp_delta_cmd_elevator,
                        state_variables=self.state_variables,
                        target=0.0,
                        potential_difference_based=False,
                        scaling_factor=ELEVATOR_CMD_TRAVEL_SCALING_FACTOR,
                        weight=1),
        )
        return base_components

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[prp.steps_left] <= 0
        state_quality = sim[self.last_assessment_reward]
        state_out_of_bounds = state_quality < self.MIN_STATE_QUALITY  # TODO:: issues if sequential?
        return (
            terminal_step
            or state_out_of_bounds
            or self._glide_out_of_bounds(sim)
            or self._altitude_out_of_bounds(sim)
        )

    def _glide_out_of_bounds(self, sim: Simulation) -> bool:
        glidePathAngle_error_deg = sim[self.prop_glide_angle_error_deg]
        return False    #TODO: this shall be evaluated
        return (abs(glidePathAngle_error_deg) > self.MAX_GLIDE_ANGLE_DEVIATION_DEG)


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

class SteadyGlideTask(FlightTask):
    """
    A task in which the agent shall maintain a
    - steady glide angle (adjustable)
    - steady banking angle (adjustable)
    """
    TARGET_GLIDE_ANGLE_DEG = -6   #TODO: this is arbitrary
    TARGET_ROLL_ANGLE_DEG  = -10
    DEFAULT_EPISODE_TIME_S = 60.  #TODO: make configurable
    GLIDE_ANGLE_ERROR_SCALING_DEG = 8
    ROLL_ANGLE_ERROR_SCALING_DEG = 8
    MIN_STATE_QUALITY = 0.0  # terminate if state 'quality' is less than this
    MAX_GLIDE_ANGLE_DEVIATION_DEG = 45
    MAX_ROLL_ANGLE_DEVIATION_DEG = 90
    #those are additional properties going into the sim[] object
    prop_glideAngle_error_deg = BoundedProperty('error/glideAngle-error-deg',
                                      'error to desired glide angle [deg]', -180, 180)
    prop_rollAngle_error_deg = BoundedProperty('error/rollAngle-error-deg',
                                        'error to desired roll/banking angle [deg]', -180, 180)

    action_variables = (prp.elevator_cmd, prp.aileron_cmd)

    def __init__(self, shaping_type: Shaping, step_frequency_hz: float, aircraft: Aircraft,
                 episode_time_s: float = DEFAULT_EPISODE_TIME_S, positive_rewards: bool = True):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.max_time_s = episode_time_s
        self.step_frequency_hz = step_frequency_hz
        self.aircraft = aircraft
        self.extra_state_variables = (  self.prop_glideAngle_error_deg    #13
                                      , self.prop_rollAngle_error_deg     #14
                                      , prp.steps_left               #15
                                      , prp.flight_path_deg          #16
                                      , prp.roll_deg                 #17
                                      , prp.indicated_airspeed       #18
                                      , prp.true_airspeed            #19
                                      , prp.setpoint_flight_path_deg #20
                                      , prp.setpoint_roll_angle_deg  #21
                                      , prp.angleOfAttack_deg        #22
                                      )
        self.state_variables = FlightTask.base_state_variables + self.extra_state_variables
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)

        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.setpoints: Dict[Property, float] = {
                  prp.episode_steps: episode_steps
                , prp.setpoint_flight_path_deg: self.TARGET_GLIDE_ANGLE_DEG
                , prp.setpoint_roll_angle_deg:  self.TARGET_ROLL_ANGLE_DEG
            }
        self.inital_attitude: Dict[Property, float] = {
                  prp.initial_u_fps: self.aircraft.get_cruise_speed_fps()*0.8    #forward speed
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
        base_components = (
            rewards.AsymptoticErrorComponent(name='glideAngle_error',
                                             prop=self.prop_glideAngle_error_deg,
                                             state_variables=self.state_variables,
                                             target=0.0,
                                             is_potential_based=False,
                                             scaling_factor=self.GLIDE_ANGLE_ERROR_SCALING_DEG),
            rewards.AsymptoticErrorComponent(name='rollAngle_error',
                                             prop=self.prop_rollAngle_error_deg,
                                             state_variables=self.state_variables,
                                             target=0.0,
                                             is_potential_based=False,
                                             scaling_factor=self.ROLL_ANGLE_ERROR_SCALING_DEG),
            # add an airspeed error relative to cruise speed component?
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
        INITIAL_ALTITUDE_FT = 6000
        extra_conditions = {prp.initial_u_fps: self.inital_attitude[prp.initial_u_fps],    #forward speed
                            prp.initial_v_fps: 0,   # side component of speed; shall be 0 in steady flight
                            prp.initial_w_fps: 0,   # down component of speed; shall be 0 in steady flight
                            prp.initial_p_radps: 0, # angular speed roll
                            prp.initial_q_radps: 0,
                            prp.initial_r_radps: 0,
                            # prp.initial_roc_fpm: 0,   #overridden by flight_path
                            prp.initial_heading_deg: 0,
                            prp.initial_flight_path_deg: self.inital_attitude[prp.initial_flight_path_deg],  #to change the initial flightpath angle, change also the sink speed and the pitch
                            prp.initial_roll_deg: self.inital_attitude[prp.initial_roll_deg],
                            prp.initial_altitude_ft: INITIAL_ALTITUDE_FT,  #overrides value from tasks.py
                            prp.initial_aoa_deg: self.inital_attitude[prp.initial_aoa_deg]
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

    def set_initial_ac_attitude(self, path_angle_gamma_deg, roll_angle_phi_deg, fwd_speed_KAS = None, aoa_deg = 1.0):
        """
        Sets the initial flight path angle, the roll angle and the cruise speed to be used as initial conditions
        """
        if fwd_speed_KAS == None:
            fwd_speed_fps = self.aircraft.get_cruise_speed_fps()*0.8
        else:
            fwd_speed_fps = 1.6878099110965*fwd_speed_KAS  # from knots to fps
        self.inital_attitude: Dict[Property, float] = {
                  prp.initial_u_fps: fwd_speed_fps
                , prp.initial_flight_path_deg: path_angle_gamma_deg
                , prp.initial_roll_deg: roll_angle_phi_deg
                , prp.initial_aoa_deg: aoa_deg
        }

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._update_glideAngle_error(sim)
        self._update_rollAngle_error(sim)
        self._decrement_steps_left(sim)

    def _update_glideAngle_error(self, sim: Simulation):
        target_glideAngle_deg = sim[prp.setpoint_flight_path_deg]
        error_deg = utils.reduce_reflex_angle_deg(sim[prp.flight_path_deg] - target_glideAngle_deg)
        sim[self.prop_glideAngle_error_deg] = error_deg

    def _update_rollAngle_error(self, sim: Simulation):
        target_rollAngle_deg = sim[prp.setpoint_roll_angle_deg]
        error_deg = utils.reduce_reflex_angle_deg(sim[prp.roll_deg] - target_rollAngle_deg)
        sim[self.prop_rollAngle_error_deg] = error_deg

    def _decrement_steps_left(self, sim: Simulation):
        sim[prp.steps_left] -= 1
        # if(sim[prp.steps_left] == 900):
        #     sim[prp.setpoint_roll_angle_deg] = 30
        #     sim[prp.setpoint_flight_path_deg] = -3
        # if(sim[prp.steps_left] == 600):
        #     sim[prp.setpoint_roll_angle_deg] = -30
        #     sim[prp.setpoint_flight_path_deg] = -15
        # if(sim[prp.steps_left] == 300):
        #     sim[prp.setpoint_roll_angle_deg] = 0
        #     sim[prp.setpoint_flight_path_deg] = -6


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
        glideAngle_error_deg = sim[self.prop_glideAngle_error_deg]
        rollAngle_error_deg = sim[self.prop_rollAngle_error_deg]
        return False #TODO: Das muss ich prüfen
        return (abs(glideAngle_error_deg) > self.MAX_GLIDE_ANGLE_DEVIATION_DEG) or \
               (abs(rollAngle_error_deg) > self.MAX_ROLL_ANGLE_DEVIATION_DEG)

    #TODO: prevent negative altitudes and terminate before severe crashes happen as this may lead to NaN in state
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

        # start with engine off instead of running
        # sim.start_engines()
        # sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)

        sim.raise_landing_gear()
        self._store_reward(RewardStub(1.0, 1.0), sim)

        sim[prp.steps_left] = self.setpoints[prp.episode_steps]
        for prop, value in self.setpoints.items():
            sim[prop] = value   #update the setpoints in the simulation model



    def get_props_to_output(self) -> Tuple:
        #TODO: this shall go into a graph or better to go additionally to a graph
        return (prp.u_fps,
                prp.flight_path_deg, prp.setpoint_flight_path_deg, self.prop_glideAngle_error_deg,
                prp.roll_deg, prp.setpoint_roll_angle_deg, self.prop_rollAngle_error_deg,
                self.last_agent_reward, self.last_assessment_reward, prp.steps_left)

    def get_timeline_props_to_output(self) -> Tuple:
        return (prp.flight_path_deg, self.prop_glideAngle_error_deg, prp.elevator,
                prp.roll_deg, self.prop_rollAngle_error_deg, prp.aileron_cmd,)

class SteadyRollAngleTask(SteadyGlideTask):
    """
    A task in which the agent shall maintain a
    - steady banking angle (adjustable)

    The only difference to the SteadyGlideTask is that the glide path angle does not contribute to the reward.
    The glide path angle error is calculated as well as this is fed into the PID controller for elevator control.
    """
    THROTTLE_CMD = 0.8          #TODO: in glide descent with engine off, this is not set
    MIXTURE_CMD = 0.8           #TODO: in glide descent with engine off, this is not set
    #TODO, these values are not in effect in the roll angle task, as the class variables of the parent class are used.
    TARGET_GLIDE_ANGLE_DEG = -6   # this is a steeper descent than in SteadyGlideTask to hold for bigger roll angles
    TARGET_ROLL_ANGLE_DEG  = -10
    DEFAULT_EPISODE_TIME_S = 120.#TODO: make configurable
    ROLL_ANGLE_ERROR_SCALING_DEG = 8
    MIN_STATE_QUALITY = 0.0  # terminate if state 'quality' is less than this
    MAX_ROLL_ANGLE_DEVIATION_DEG = 60
    MIN_ALTITUDE = 200 # the minimum allowed altitude; This is checked to avoid severe crashes during training which my lead to NaN in the state
    glideAngle_error_deg = BoundedProperty('error/glideAngle-error-deg',
                                      'error to desired glide angle [deg]', -180, 180)
    rollAngle_error_deg = BoundedProperty('error/rollAngle-error-deg',
                                        'error to desired roll/banking angle [deg]', -180, 180)
    action_variables = (prp.elevator_cmd, prp.aileron_cmd)  #the elevator_cmd is in for the PID controller

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        base_components = (
            rewards.AsymptoticErrorComponent(name='rollAngle_error',
                                             prop=self.rollAngle_error_deg,
                                             state_variables=self.state_variables,
                                             target=0.0,
                                             is_potential_based=False,
                                             scaling_factor=self.ROLL_ANGLE_ERROR_SCALING_DEG),
        )
        return base_components


    def _decrement_steps_left(self, sim: Simulation):
        sim[prp.steps_left] -= 1
        # if(sim[prp.steps_left] == 900):
        #     sim[prp.setpoint_roll_angle_deg] = 30
        #     sim[prp.setpoint_flight_path_deg] = -3
        # if(sim[prp.steps_left] == 600):
        #     sim[prp.setpoint_roll_angle_deg] = -30
        #     sim[prp.setpoint_flight_path_deg] = -15
        # if(sim[prp.steps_left] == 300):
        #     sim[self.target_rollAngle_deg] = 0
        #     sim[prp.setpoint_flight_path_deg] = -6


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
        rollAngle_error_deg = sim[self.rollAngle_error_deg]
        return False
        return (abs(rollAngle_error_deg) > self.MAX_ROLL_ANGLE_DEVIATION_DEG)

    def _altitude_out_of_bounds(self, sim: Simulation) -> bool:
        altitude = sim[prp.altitude_sl_ft]
        if (altitude <= self.MIN_ALTITUDE):
            print("altitude too small: {}; aborting".format(altitude))
            return True
        else:
            return False


    def _new_episode_init(self, sim: Simulation) -> None:
        # no need to reimplement
        super()._new_episode_init(sim)


    def get_props_to_output(self) -> Tuple:
        """
        This is only a list of Property values, not the values itself.
        The values itself are then queried from the Simulation object
        """
        #TODO: this shall go into a graph or better to go additionally to a graph
        return (prp.u_fps,
                prp.flight_path_deg, prp.setpoint_flight_path_deg, self.glideAngle_error_deg,
                prp.roll_deg, prp.setpoint_roll_angle_deg, self.rollAngle_error_deg,
                self.last_agent_reward, self.last_assessment_reward, self.steps_left)

    def get_timeline_props_to_output(self) -> Tuple:
        """
        This is only a list of Property values, not the values itself.
        The values itself are then queried from the Simulation object
        """
        return (prp.flight_path_deg, self.glideAngle_error_deg, prp.elevator,
                prp.roll_deg, self.rollAngle_error_deg, prp.aileron_cmd,)



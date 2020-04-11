import numpy as np
import gym
import random
import types
from typing import List, Dict, Tuple, Callable, Optional

from collections import namedtuple
from abc import ABC, abstractmethod

import gym_jsbsim.properties as prp

from gym_jsbsim.simulation import Simulation
from gym_jsbsim.properties import BoundedProperty
from gym_jsbsim.aircraft import Aircraft

from gym_jsbsim.environment_eee import JsbSimEnv_multi_agent


class AgentTask(ABC):
    """
    Interface for AgentTasks, which implement the reward calculation and state observation in a multi-agent environment.

    An AgentTask defines its observation space, custom properties, setpoints, action space, termination conditions and agent_reward function.

    A List of AgentTasks is injected into a Multi-Agent environment to handle the observations, the rewards and the done flags.
    """
    def __init__(self, name: str):
        """ Each Agent Taskt needs those porperty lists initialized to be queried.

        The content of the lists must be set for each AgentTask individually.
        Each list entry shall be of type BoundedProperty (TODO: Do I need to define all of them in the properties.py file, or can there be local ones as well?)
        """
        self.name = name                                    # name of the agent, used for the naming of properties
        self.obs_props: List[BoundedProperty] = []          # properties returned to the Agent as observation. Either directly from JSBSim or from custom_props
        self.custom_props: List[BoundedProperty] = []       # properties calculated by the AgentTask. May or may not be part of the obs_props
        self.setpoints: Dict[BoundedProperty, float] = {}   # setpoints to use in the error/deviation calculation. May be (dynamically) changed in the course of the simulation. Stored into the sim-object
        self.action_props: List[BoundedProperty] = []        # actions issued to JSBSim in each step from the associated Agent. Informative for AgentTask to e. g. incorporate it into reward calculation

    def inject_environment(self, env: JsbSimEnv_multi_agent):
        """ Injects the environment, the AgentTask is acting in.
        Mostly used to have access to the env.sim object for data storage and retrieval.
        
        It's really easier to consider the AgentTasks as part of the environment with access to the sim object.
        """
        self.env = env
        self.sim = self.env.sim #for caching the simulator object used as data storage
        self.dt  = self.env.dt  #for caching the step-time to calculate the integral
        #store the setpoints to the sim object
        for sp_prop, sp_val in self.setpoints.items():
            self.sim[sp_prop] =sp_val

    
    
    def calculate(self) -> Tuple[float, bool, Dict]:
        """ Calculate the task specific reward from the actual observation and 
        checks end of episode wrt. the specific AgentTask. Additional info is 
        also determined (reward components).

        The reward is a function of the actual observation and -if necessary- 
        the last observation (for potential based rewards).

        Each task calculates its own reward components. Hence, each task may follow 
        individual targets. May it be collaborative with other tasks or competitive.

        Each AgentTask may have individual termination conditions wrt. its own observations. 
        If one AgentTask detects the end of an episode, the episode for all agents must terminate
        """
        done = self._check_end_of_episode()
        rwd, rwd_components = self._calculate_reward()

        return (rwd, done, {'reward_components': rwd_components})

    def _check_for_done(self) -> bool:
        """
        Checks if the episode shall end due to any condition (e. g. properties is out of bounds.)

        Shall be overridden in inherited classes or injected in individual instances as method.

        :return: True if values are out of bounds and hence the episode should end.
        """
        return False
    
    def _calculate_reward(self) -> Tuple[int, Dict[str, float]]:
        raise NotImplementedError(self.__class__.__name__+'._calculate_reward(self) shall be implemented in subclass or injected into the instance.')

    def get_state_space(self) -> gym.Space:
        """ Get the task's state/observation space object. 
        
        Returns the observation space, the AgentTask operates on.

        Naming is in line with Go-Ren's JsbSimEnv.
        """
        state_lows = np.array([state_var.min for state_var in self.obs_props])
        state_highs = np.array([state_var.max for state_var in self.obs_props])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self) -> gym.Space:
        """ Get the task's action Space object """
        action_lows = np.array([act_prop.min for act_prop in self.action_props])
        action_highs = np.array([act_prop.max for act_prop in self.action_props])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')
    
    def change_setpoints(self, new_setpoints: Dict[BoundedProperty, float]):
        """
        Changes the setpoints for the AgentTask. The changes will take effect within the next environment step. (call to env.step())
        The setpoint values are stored within a property in the env's sim object.

        Implement _change_setpoint_helper() in derived classes to implement associated actions (e. g. integrals must be reset when changing the setpoints)

        :param new_setpoints: A dictionary with new setpoints to be used. New values overwrite old ones.
        """
        for prop, value in new_setpoints.items():
            self._change_setpoint_helper( (prop, value) )    #implemented in subclass
            self.setpoints[prop] = value    #update the setpoints in the AgentTask class
            #TODO: storing the setpoints to the sim object is delegated to the _change_setpoint_helper() for the moment
                
    @abstractmethod
    def _change_setpoint_helper(self, changed_setpoint: Tuple[BoundedProperty,float]):
        """ 
        Any actions regarding setpoint changes specific for the special task are implemented here. 
        If needed, the setpoint storage to sim onject also happens here.

        Called for each changed setpoint.

        Reset Integrals, notify  dependant objects (e. g. by callback), store the new setpoint to the sim object...
        """
        raise NotImplementedError('_change_setpoint_helper() must be imlemented in '+self.__class__+'. (Maybe just a "pass"-statement).')
    
    def get_setpoints(self) -> Dict[BoundedProperty, float]:
        """ just returns the setpoints of the AgentTask.
        """
        return self.setpoints

    @abstractmethod
    def initialize_custom_properties(self):
        """
        Initializes all custom properties after a reset() to the environment.
        This includes staes and controls.

        Called on every AgentTask from env.reset()
        """
        pass

    @abstractmethod
    def update_custom_properties(self, action: np.ndarray):
        """ Updates state elements (custom properties) within the AgentTask's own responsibility.

        Is called from within the env.step() function for each AgentTask taking part in the environemnt. 
        Is called after the simulator did its work before the individual rewards and dones are calculated.

        Only properties depending on locally known values (e. g. the last action) may be part of the custom properties.
        No properties with interdependencies between different AgentTasks may be part of these custom_properties 
        as the calculation order is undefined. However, it is possible to include "foreign" custom props in the 
        observation space of an AgentTask.
        
        :param action: the actions issued by the associated Agent. May or may not be used to be incorporated in custom_props 
           (and later on in the reward calculation)
        """
        raise NotImplementedError('update_custom_properties() must be imlemented in '+self.__class__+'.')

    @abstractmethod
    def get_props_to_output(self) -> List[prp.Property]:
        """
        Provides a list or properties to be collected by the EpisodePlotter wrapper to include in an episode plot.

        By means of this method, any AgentTask can provide a whishlist of plotted props without adding it to the state.

        """
        return []


class FlightAgentTask(AgentTask):   #implements the same interface like Go-Ren's Task, with the needed adapations to multi-agent setting
    def __init__(self, name):
        super(FlightAgentTask, self).__init__(name)
        pass





from gym_jsbsim.utils import reduce_reflex_angle_deg
class PID_FlightAgentTask(FlightAgentTask):
    """ A class to implement a PID controller for a certain actuation.

    The class PID_FlightAgentTask takes the value to be controlled as an input state. 

    The PID_FlightAgentTask adds the difference from the input state value to the 
    setpoint value to the custom properties. 

    The reward calculated is always the negative absolute difference of the controlled value to the setpoint. 
    The reward is _not_ normalized in any way and will be negative. (It is not used for any training!)
    """

    def __init__(self, name: str, actuating_prop: BoundedProperty, setpoint: Dict[BoundedProperty, float], 
                change_setpoint_callback: Callable[[float], None] = None, measurement_in_degrees = True, max_allowed_error = 30):
        """
        :param actuating_prop: The actuation variable to be controlled
        :param setpoint: The setpoint property to be used for deviation calculation
        :param change_setpoint_callback: For the PID_FlightAgentTask, the Agent should pass in a callback 
            function to be notified on setpoint changes. This callback can reset the PID internal integrators.
        :param measurement_in_degrees: indicates if the controlled property and the setpoint is given in degrees
        :param max_allowed_error = 30: The maximum absolute error, before an episode ends. Be careful with setpoint changes! Can be set to None to disable checking.
        """
        super(PID_FlightAgentTask, self).__init__(name)

        self.measurement_in_degrees = measurement_in_degrees
        self.max_allowed_error = max_allowed_error

        #custom properties 
        self.prop_error = BoundedProperty('error/'+self.name+'_err', 'error to desired setpoint', -180, 180)
        self.prop_error_integral = BoundedProperty('error/'+self.name+'_int', 'integral of the error multiplied by timestep', -float('inf'), +float('inf'))
        self.prop_delta_cmd = BoundedProperty('info/'+self.name+'_delta-cmd', 'the actuator travel/movement since the last step', 
                actuating_prop.min - actuating_prop.max, actuating_prop.max - actuating_prop.min)
        self.prop_setpoint = BoundedProperty('setpoint/'+self.name+'_setpoint', 'the setpoint for the '+self.name+' controller', -180, 180)

        self.actuating_prop = actuating_prop
        self.last_action = 0
        self.setpoint_prop, self.setpoint_value = list(setpoint.items())[0] #there's only one setpoint for the PID controller
            #TODO: the setpoint is now stored twice. Not really useful. Do I need this caching? Do I need the self.setpoints-dictionary?
        
        self.obs_props = [self.prop_error]
        self.custom_props = [self.prop_error, self.prop_error_integral, self.prop_delta_cmd, self.prop_setpoint]
        self.action_props = [actuating_prop]

        # the value of the setpoint_prop itself is not really relevant, it must be in the simulator object to be retrieved, but not passed anywhere
        # if not all([setpoint_prop in (self.obs_props + self.custom_props) for setpoint_prop in setpoint.keys()]):
        #     raise ValueError('All setpoints must match a property in obs_props or in custom_props')

        self.change_setpoint_callback = change_setpoint_callback    #to notify the PID Agent that there is a new setpoint in effect
        self.change_setpoints(setpoint) #TODO: How to reset the Agent's internal integrator Use the info field? No, it's too late for one step then. Add a special function to the PID-Agent!!!

    def _change_setpoint_helper(self, changed_setpoint:Tuple[BoundedProperty,float]):
        self.setpoint_prop, self.setpoint_value = changed_setpoint
        if self.change_setpoint_callback:
            self.change_setpoint_callback(self.setpoint_value)
        
    def update_custom_properties(self, action: np.ndarray):
        error = self.sim[self.setpoint_prop] - self.setpoint_value
        if self.measurement_in_degrees:
            error = reduce_reflex_angle_deg(error)
        self.sim[self.prop_error] = error
        self.sim[self.prop_error_integral] += error * self.dt
        self.sim[self.prop_delta_cmd] = action[0] - self.last_action
        self.sim[self.prop_setpoint] = self.setpoint_value 
        self.last_action = action[0]

    def initialize_custom_properties(self):
        """ Initializes all the custom properties to start values

        TODO: check if this can integrated with update_custom_properties
        """
        error = self.sim[self.setpoint_prop] - self.setpoint_value
        if self.measurement_in_degrees:
            error = reduce_reflex_angle_deg(error)
        self.sim[self.prop_error] = error
        self.sim[self.prop_error_integral] = error * self.dt    #TODO: is it correct to add the initial error to the integral or shold it be added just _after_ the next timestep
        self.sim[self.prop_delta_cmd] = 0
        self.sim[self.prop_setpoint] = self.setpoint_value 
        self.last_action = self.sim[self.actuating_prop]

    def calculate(self) -> Tuple[float, bool, Dict]:
        done = self._check_values_out_of_bounds()
        rwd = -abs(self.sim[self.prop_error])     #this reward is negative and not normailzed. It is not suitable for training an RL agent, but it may be OK to compare algorithms
        #TODO: why not create a reward measure to compare the quality of control algos in the testing phase. Include Overshoot oscillation and so on....
        info = {'reward': rwd, 'reward_components': {self.name+'_err_rwd': rwd}}

        return (rwd, done, info) 

    def _check_values_out_of_bounds(self) -> bool:
        """
        Checks if any of the value in observed state or custom properties is out of bounds.

        :return: True if values are out of bounds and hence the episode should end.
        """
        if self.max_allowed_error:  
            return abs(self.sim[self.prop_error]) >= self.max_allowed_error
        else:
            return False
    
    def get_props_to_output(self) -> List[prp.Property]:
        output_props = self.custom_props + [
            self.setpoint_prop
        ]
        return output_props

# class DDPG_FlightAgentTask(FlightAgentTask):
#     """ A class to implement a DDPG based controller for a certain actuation.

#     The class DDPG_FlightAgentTask takes the value to be controlled as an input state. 

#     The DDPG_FlightAgentTask calculates the error wrt. the setpoint and some limited error integral to the custom properties. 
#     Additionally, the current action and the last actuator travel (Δ-Value) are added to the custom properties and the observation.

#     The reward calculation takes ... TODO:
#     """

#     def __init__(self, name: str, actuating_prop: BoundedProperty,  setpoint: Dict[BoundedProperty, float], 
#                  observation_props: List[BoundedProperty], 
#                  reward_function: Callable[...,[Tuple[float, Dict[str, float]]]] = None, 
#                  out_of_bounds_check_function: Callable[...,[bool]] = None, 
#                  measurement_in_degrees = True, max_allowed_error = 30):
#         """
#         :param actuating_prop: The actuation variable to be controlled
#         :param setpoint: The setpoint property to be used for deviation calculation
#         :param observation_props: The state properties that shall be presented to the DDPG Agent in the observation. Custom-Props
#         :param measurement_in_degrees: indicates if the controlled property and the setpoint is given in degrees
#         :param max_allowed_error = 30: The maximum absolute error, before an episode ends. Be careful with setpoint changes! Can be set to None to disable checking.
#         """
#         super(DDPG_FlightAgentTask, self).__init__(name)

#         # inject the reward calculation and the values out of bounds check into the instance like explained here
#         # https://tryolabs.com/blog/2013/07/05/run-time-method-patching-python/ and here
#         # https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance#comment66379065_2982 
#         if reward_function:
#             self._calc_reward = reward_function.__get__(self)
#         if out_of_bounds_check_function:
#             self._check_for_done = out_of_bounds_check_function.__get__(self)

#         self.measurement_in_degrees = measurement_in_degrees
#         self.max_allowed_error = max_allowed_error
#         self.prop_error = BoundedProperty('error/'+self.name+'_err', 'error to desired setpoint', -180, 180)
#         self.prop_error_integral = BoundedProperty('error/'+self.name+'_int', 'integral of the error multiplied by timestep', -float('inf'), +float('inf'))
#         self.prop_delta_cmd = BoundedProperty('info/'+self.name+'_delta-cmd', 'the actuator travel/movement since the last step', 
#                 actuating_prop.min - actuating_prop.max, actuating_prop.max - actuating_prop.min)
#         self.prop_setpoint = BoundedProperty('setpoint/'+self.name+'_setpoint', 'the setpoint for the '+self.name+' controller', -180, 180)

#         self.actuating_prop = actuating_prop
#         self.last_action = 0
#         self.setpoint_prop, self.setpoint_value = list(setpoint.items())[0] #there's only one setpoint for the PID controller
#             #TODO: the setpoint is now stored twice. Not really useful. Do I need this caching? Do I need the self.setpoints-dictionary?
        
#         self.obs_props = presented_state + [self.prop_error, self.prop_error_integral, self.prop_delta_cmd]
#         self.custom_props = [self.prop_error, self.prop_error_integral, self.prop_delta_cmd, self.prop_setpoint]
#         self.action_props = [actuating_prop]

#         self.change_setpoints(setpoint)

#     def _change_setpoint_helper(self, changed_setpoint:Tuple[BoundedProperty,float]):
#         self.setpoint_prop, self.setpoint_value = changed_setpoint
#         if self.change_setpoint_callback:
#             self.change_setpoint_callback(self.setpoint_value)
        
#     def update_custom_properties(self, action: np.ndarray):
#         error = self.sim[self.setpoint_prop] - self.setpoint_value
#         if self.measurement_in_degrees:
#             error = reduce_reflex_angle_deg(error)
#         self.sim[self.prop_error] = error
#         self.sim[self.prop_error_integral] += error * self.dt
#         self.sim[self.prop_delta_cmd] = action[0] - self.last_action
#         self.sim[self.prop_setpoint] = self.setpoint_value 
#         self.last_action = action[0]

#     def initialize_custom_properties(self):
#         """ Initializes all the custom properties to start values

#         TODO: check if this can integrated with update_custom_properties
#         """
#         error = self.sim[self.setpoint_prop] - self.setpoint_value
#         if self.measurement_in_degrees:
#             error = reduce_reflex_angle_deg(error)
#         self.sim[self.prop_error] = error
#         self.sim[self.prop_error_integral] = error * self.dt    #TODO: is it correct to add the initial error to the integral or shold it be added just _after_ the next timestep
#         self.sim[self.prop_delta_cmd] = 0
#         self.sim[self.prop_setpoint] = self.setpoint_value 
#         self.last_action = self.sim[self.actuating_prop]
    
#     def get_props_to_output(self) -> List[prp.Property]:
#         output_props = self.custom_props + [
#             self.setpoint_prop
#         ]
#         return output_props
            

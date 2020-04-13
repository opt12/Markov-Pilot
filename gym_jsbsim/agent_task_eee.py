import numpy as np
import gym
import random
import types
from typing import List, Dict, Tuple, Callable, Optional

from collections import namedtuple
from abc import ABC, abstractmethod

import gym_jsbsim.properties as prp
from gym_jsbsim import assessors, rewards

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
    def __init__(self, name: str, 
                 make_base_reward_components: Callable[['AgentTask'], Tuple[rewards.RewardComponent, ...]] = None,
                 is_done: Callable[['AgentTask'], bool] = None):
        """ Each Agent Taskt needs those porperty lists initialized to be queried.

        The content of the lists must be set for each AgentTask individually.
        Each list entry shall be of type BoundedProperty (TODO: Do I need to define all of them in the properties.py file, or can there be local ones as well?)
        """
        self.name = name                                    # name of the agent, used for the naming of properties
        self.obs_props: List[BoundedProperty] = []          # properties returned to the Agent as observation. Either directly from JSBSim or from custom_props
        self.custom_props: List[BoundedProperty] = []       # properties calculated by the AgentTask. May or may not be part of the obs_props
        self.setpoint_props = ()                            # the properties assigned to the setpoints; subtract the value of setpoint prop from the base prop to calculate the error
        self.initial_setpoint_values = ()                   # the initial setpoint values, used to store values to sim object, don't get updated on hange_setpoint
        self.setpoint_value_props = []                      # setpoints to use in the error/deviation calculation. May be (dynamically) changed in the course of the simulation. Stored into the sim-object
        self.action_props: List[BoundedProperty] = []       # actions issued to JSBSim in each step from the associated Agent. Informative for AgentTask to e. g. incorporate it into reward calculation
        self.positive_rewards = True                        # determines whether only positive rewards are possible (see Gor-Ren's documentation)

        if make_base_reward_components:
            self._make_base_reward_components = make_base_reward_components.__get__(self)   # bind the injected function to the instance
        #TODO: set assessor to some NullAssessor raising an exception explaining what went wrong
        self.assessor = None                                # call self.assessor = self._make_assessor() in your constructor after preparing all properties

        if is_done:
            self._is_done = is_done.__get__(self)            # bind the injected function to the instance

    def _make_assessor(self):
        """
        Returns an Assessor Object to evaluate the value of a state in the context of the AgentTask

        In contrast to Gor-Ren's original implementation, only the STANDARD model 
        is supported with no reward shaping nor sequential rewards. (If needed in the future, this
        may be overridden in the future)

        The function 
        _make_base_reward_components(self)
        shall be injected into the concrete AgentTask at construction time

        """
        while True:
            try:
                base_components = self._make_base_reward_components()
            except rewards.RewardNotVisibleError as e:
                #There is a property misssing in the presented state_variables, so we add it to self.obs_props:
                if e.prop in self.obs_props:
                    """
                    you _must_ present 
                            state_variables=self.obs_props
                    in the reward components. Otherwise there is no point in adding them to self.obs_props
                    """
                    raise ValueError(f'{e.prop}  is not in list.')

                self.obs_props.append(e.prop)
                print(f"AgentTask: {self.name}: Added property {e.prop} to self.obs_props.")
                continue
            break


        return assessors.AssessorImpl(base_components, (), positive_rewards=self.positive_rewards)
    
    def inject_environment(self, env: JsbSimEnv_multi_agent):
        """ Injects the environment, the AgentTask is acting in.
        Mostly used to have access to the env.sim object for data storage and retrieval.
        
        It's really easier to consider the AgentTasks as part of the environment with access to the sim object.
        """
        self.env = env
        self.sim = self.env.sim #for caching the simulator object used as data storage
        self.dt  = self.env.dt  #for caching the step-time to calculate the integral
        #store the initial_setpoints to the sim object
        for sp, val in zip(self.setpoint_value_props, self.initial_setpoint_values):
            self.sim[sp] = val
    
    def assess(self, obs, last_obs) -> Tuple[float, bool, Dict]:
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
        done = self._is_done()
        rwd, rwd_components = self.assessor.assess(obs, last_obs, done)

        return (rwd, done, {'reward_components': rwd_components})
    
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
    
    @abstractmethod
    def change_setpoints(self, new_setpoints: Dict[BoundedProperty, float]):
        """
        Changes the setpoints for the AgentTask. The changes will take effect within the next environment step. (call to env.step())
        The setpoint values are stored within a property in the env's sim object.

        If needed, clean-up actions shall be performed here (like e. g. reset integrators)

        :param new_setpoints: A dictionary with new setpoints to be used. New values overwrite old ones.
        """
        pass
                
    def get_setpoint_props(self) -> Dict[BoundedProperty, float]:
        """ just returns the props with setpoints for the AgentTask
        """
        return self.setpoint_value_props

    def print_info(self):
        """
        Prints out all relevant information on the TaskAgent
        """
        print("********************************************")
        print(f"TaskAgent '{self.name}':")
        obs_props_list = [prop.name for prop in self.obs_props]
        print(f"obs_props[{len(self.obs_props)}]:", end="")
        print(*obs_props_list, sep = ", ")
        print(f"Observation Space:\n", self.get_state_space(), "\nlow:  ", self.get_state_space().low, "\nhigh: ", self.get_state_space().high, sep="")

        custom_props_list = [prop.name for prop in self.custom_props]
        print(f"custom_props[{len(self.custom_props)}]:", end="")
        print(*custom_props_list, sep = ", ")

        action_props_list = [prop.name for prop in self.action_props]
        print(f"action_props[{len(self.action_props)}]:", end="")
        print(*action_props_list, sep = ", ")
        print(f"Action Space: ", self.get_action_space(), "\nlow:  ", self.get_action_space().low, "\nhigh: ", self.get_action_space().high, sep="")

        print("********************************************")

    @abstractmethod
    def _make_base_reward_components(self):
        # pylint: disable=method-hidden
        """ Defines the components used in the state assessment.
        
        This function shall be injected into the AgentTask at construction time as it is 
        individual for each of them. The injected function is bound to the instantiated object 
        and hence has access to all instance variables.
        This seems to be more flexible than subclassing.

        Alternatively, subclassing and overwriting _make_base_reward_components() is also possible.
        """
        raise NotImplementedError('_make_base_reward_components() must be injected into '+self.__class__+' at instantiation time.')

    @abstractmethod
    def _is_done(self) -> bool:
        # pylint: disable=method-hidden
        """
        Checks if the episode shall end due to any condition (e. g. properties is out of bounds.)

        Each AgentTask may have individual termination conditions wrt. its own observations. 
        If one AgentTask detects the end of an episode, the episode for all agents must terminate

        Shall be overridden in inherited classes or injected in individual instances as method.

        :return: True if values are out of bounds and hence the episode should end.
        """
        return False

    # @abstractmethod
    # def _change_setpoint_helper(self, changed_setpoint: Tuple[BoundedProperty,float]):
    #     """ 
    #     Any actions regarding setpoint changes specific for the special task are implemented here. 
    #     If needed, the setpoint storage to sim onject also happens here.

    #     Called for each changed setpoint.

    #     Reset Integrals, notify  dependant objects (e. g. by callback), store the new setpoint to the sim object...
    #     """
    #     raise NotImplementedError('_change_setpoint_helper() must be imlemented in '+self.__class__+'. (Maybe just a "pass"-statement).')
    
    @abstractmethod
    def initialize_custom_properties(self):
        """
        Initializes all custom properties after a reset() to the environment.
        This includes staes and controls.

        Called on every AgentTask from env.reset()
        """
        pass

    @abstractmethod
    def update_custom_properties(self):
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

#TODO: get rid of this intermediate superclass. Why do we need it?
# class FlightAgentTask(AgentTask):   #implements the same interface like Go-Ren's Task, with the needed adapations to multi-agent setting
#     def __init__(self, name):
#         super(FlightAgentTask, self).__init__(name)
#         pass


from gym_jsbsim.utils import reduce_reflex_angle_deg
'''
class PID_FlightAgentTask(AgentTask):
    """ A class to implement a PID controller for a certain actuation.

    The class PID_FlightAgentTask takes the value to be controlled as an input state. 

    The PID_FlightAgentTask adds the difference from the input state value to the 
    setpoint value to the custom properties. 

    The reward calculated is always the negative absolute difference of the controlled value to the setpoint. 
    The reward is _not_ normalized in any way and will be negative. (It is not used for any training!)
    """

    def __init__(self, name: str, actuating_prop: BoundedProperty, setpoint: Dict[BoundedProperty, float], 
                make_base_reward_components: Callable[['AgentTask'], Tuple[rewards.RewardComponent, ...]] = None,
                is_done: Callable[['AgentTask'], bool] = None, 
                change_setpoint_callback: Callable[[float], None] = None, 
                measurement_in_degrees = True, max_allowed_error = 30):
        """

        _make_base_reward_components and _check_for_done are overridden in the subclass

        :param actuating_prop: The actuation variable to be controlled
        :param setpoint: The setpoint property to be used for deviation calculation
        :param _make_base_reward_components = None: if necessary inject a custom function to be bound to instance.
            Not needed in case of PID control as _make_base_reward_components() is overwritten by default in subclass.
        :param is_done = None: if necessary inject a custom function to be bound to instance.
            Not needed in case of PID control as _check_for_done() is overwritten by default in subclass to check for max_allowed_error.
        :param change_setpoint_callback: For the PID_FlightAgentTask, the Agent should pass in a callback 
            function to be notified on setpoint changes. This callback can reset the PID internal integrators.
        :param measurement_in_degrees: indicates if the controlled property and the setpoint is given in degrees
        :param max_allowed_error = 30: The maximum absolute error, before an episode ends. Be careful with setpoint changes! Can be set to None to disable checking.
        """
        super(PID_FlightAgentTask, self).__init__(name, 
                                                  make_base_reward_components=make_base_reward_components,
                                                  is_done=is_done)

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

        self.assessor = self._make_assessor()   #this can only be called after the preparattion of all necessary props
        self.print_info()

    def _make_base_reward_components(self):     #may be overwritten by injected custom function
        # pylint: disable=method-hidden
        """
        Just adds an Asymptotic error component as standard reward to the PID_AgentTask.

        May be overwritten by injected custom function.
        """
        ANGULAR_DEVIATION_SCALING = 0.5 #just a default value which neither fits glide nor roll angle really good
        base_components = (
            rewards.AngularAsymptoticErrorComponent(name='rwd_'+self.name+'_asymptotic_error',
                                    prop=self.prop_error,
                                    state_variables=self.obs_props,
                                    target=0.0,
                                    potential_difference_based=False,
                                    scaling_factor=ANGULAR_DEVIATION_SCALING,
                                    weight=1),
            )
        return base_components
    
    def _is_done(self):      #may be overwritten by injected custom function
        # pylint: disable=method-hidden
        """
        Checks if the observed error is out of bounds.

        :return: True if values are out of bounds and hence the episode should end.
        """
        if self.max_allowed_error:  
            return abs(self.sim[self.prop_error]) >= self.max_allowed_error
        else:
            return False
    
    def _change_setpoint_helper(self, changed_setpoint:Tuple[BoundedProperty,float]):
        self.setpoint_prop, self.setpoint_value = changed_setpoint
        if self.change_setpoint_callback:
            self.change_setpoint_callback(self.setpoint_value)
        
    def update_custom_properties(self):
        error = self.sim[self.setpoint_prop] - self.setpoint_value
        if self.measurement_in_degrees:
            error = reduce_reflex_angle_deg(error)
        self.sim[self.prop_error] = error
        self.sim[self.prop_error_integral] += error * self.dt
        self.sim[self.prop_delta_cmd] = self.sim[self.action_props[0]] - self.last_action
        self.sim[self.prop_setpoint] = self.setpoint_value 
        self.last_action = self.sim[self.action_props[0]]

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

    def get_props_to_output(self) -> List[prp.Property]:
        output_props = self.custom_props + [
            self.setpoint_prop
        ]
        return output_props
'''
class SingleChannel_FlightAgentTask(AgentTask): #TODO: check whether it would be better to call it SingleAngularChannel_FlightAgentTask(AgentTask)
    """ A class to implement a controller for a single channel actuation with a single error measurement.

    The class SingleChannel_FlightAgentTask takes the value to be controlled as an input state. 

    The SingleChannel_FlightAgentTask calculates the error wrt. the setpoint and some limited error integral to the custom properties. 
    Additionally, the current action and the last actuator travel (Δ-Value) are added to the custom properties and the observation.

    The reward uses the approach of Gor-Ren. The make_base_reward_components-Function shall be injected to the 
    AgentTask object at instantiation time.
    """

    def __init__(self, name: str, actuating_prop: BoundedProperty,  setpoints: Dict[BoundedProperty, float], 
                presented_state: List[BoundedProperty] = [], 
                make_base_reward_components: Callable[['AgentTask'], Tuple[rewards.RewardComponent, ...]] = None,
                is_done: Callable[['AgentTask'], bool] = None, 
                # change_setpoint_callback: Callable[[float], None] = None, #TODO: we used to have that, but it's not used anymore, remove
                measurement_in_degrees = True, max_allowed_error = 30,
                integral_limit = float('inf'), integral_decay = 1):
        """
        :param actuating_prop: The actuation variable to be controlled
        :param setpoints: The setpoint property to be used for deviation calculation
        :param presented_state: The additional state properties that shall be presented to the Agent besides the props defined within the AgentTask. 
        :param make_base_reward_components = None: Inject a custom function to be bound to instance.
        :param is_done = None: Inject a custom function to be bound to instance.
            Default just checks for max_allowed_error in the self.prop_error custom property.
        # :param change_setpoint_callback: For the PID_FlightAgentTask, the Agent should pass in a callback 
            function to be notified on setpoint changes. This callback can reset the PID internal integrators.
        :param measurement_in_degrees: indicates if the controlled property and the setpoint is given in degrees
        :param max_allowed_error = 30: The maximum absolute error, before an episode ends. Be careful with setpoint changes! Can be set to None to disable checking.
        :param integral_limit = float('inf'): the limiting value for the error integrator Error integral is clipped to [-integral_limit, integral_limit]
        :param integral_decay = 1: the decay factor for the integral value
        """
        super(SingleChannel_FlightAgentTask, self).__init__(name, 
                                            make_base_reward_components=make_base_reward_components,
                                            is_done=is_done)

        self.measurement_in_degrees = measurement_in_degrees
        self.max_allowed_error = max_allowed_error
        self.integral_limit = integral_limit
        self.integral_decay = integral_decay

        #custom properties 
        self.prop_error = BoundedProperty('error/'+self.name+'_err', 'error to desired setpoint', -float('inf'), float('inf'))
        self.prop_error_integral = BoundedProperty('error/'+self.name+'_int', 'integral of the error multiplied by timestep', -float('inf'), +float('inf'))
        self.prop_delta_cmd = BoundedProperty('info/'+self.name+'_delta-cmd', 'the actuator travel/movement since the last step', 
                actuating_prop.min - actuating_prop.max, actuating_prop.max - actuating_prop.min)
                
        self.actuating_prop = actuating_prop
        self.last_action = 0
        # self.setpoint_value = list(setpoint.items())[0] #there's only one setpoint for the single channel controller
            #TODO: the setpoint is now stored twice. Not really useful. Do I need this caching? Do I need the self.setpoints-dictionary?
        
        self.obs_props = [self.prop_error, self.prop_error_integral, self.prop_delta_cmd] + presented_state
        self.custom_props = [self.prop_error, self.prop_error_integral, self.prop_delta_cmd]
        self.action_props = [actuating_prop]

        # the value of the setpoint_prop itself is not really relevant, it must be in the simulator object to be retrieved, but not passed anywhere
        # if not all([prop in (self.obs_props + self.custom_props) for prop in setpoints]):
        #     raise ValueError('All setpoints must match a property in obs_props or in custom_props')

        # self.setpoints = setpoints
        self.setpoint_props, self.initial_setpoint_values = zip(*setpoints.items())    #returns immutable tuples
        self.setpoint_value_props = [sp.prefixed('setpoint') for sp in self.setpoint_props]

        # self.change_setpoint_callback = change_setpoint_callback    #to notify the PID Agent that there is a new setpoint in effect
        # self.change_setpoints(setpoints) #TODO: How to reset the Agent's internal integrator Use the info field? No, it's too late for one step then. Add a special function to the PID-Agent!!!

        self.assessor = self._make_assessor()   #this can only be called after the preparattion of all necessary props
        self.print_info()

    def _make_base_reward_components(self):     #may be overwritten by injected custom function
        # pylint: disable=method-hidden
        """
        Just adds an Asymptotic error component as standard reward to the PID_AgentTask.

        May be overwritten by injected custom function.
        """
        ANGULAR_DEVIATION_SCALING = 0.5 #just a default value which neither fits glide nor roll angle really good
        base_components = (
            rewards.AngularAsymptoticErrorComponent(name='rwd_'+self.name+'_asymptotic_error',
                                    prop=self.prop_error,
                                    state_variables=self.obs_props,
                                    target=0.0,
                                    potential_difference_based=False,
                                    scaling_factor=ANGULAR_DEVIATION_SCALING,
                                    weight=1),
            )
        return base_components
    
    def _is_done(self):      #may be overwritten by injected custom function
        # pylint: disable=method-hidden
        """
        Checks if the observed error is out of bounds.

        :return: True if values are out of bounds and hence the episode should end.
        """
        if self.max_allowed_error:  
            return abs(self.sim[self.prop_error]) >= self.max_allowed_error
        else:
            return False
    
    # def _change_setpoint_helper(self, changed_setpoint:Tuple[BoundedProperty,float]):
    #     self.setpoint_prop, self.setpoint_value = changed_setpoint
    #     if self.change_setpoint_callback:
    #         self.change_setpoint_callback(self.setpoint_value)
        
    def update_custom_properties(self):
        error = self.sim[self.setpoint_props[0]] - self.sim[self.setpoint_value_props[0]]   #only one setpoint for SingleChannel_FlightAgentTask
        if self.measurement_in_degrees:
            error = reduce_reflex_angle_deg(error)
        self.sim[self.prop_error] = error
        self.sim[self.prop_error_integral] = np.clip(    #clip the maximum amount of the integral
                        self.sim[self.prop_error_integral] * self.integral_decay + error,   #TODO: check whether a slope limit is useful as well.
                        -self.integral_limit, self.integral_limit
                    )
        self.sim[self.prop_delta_cmd] = self.sim[self.action_props[0]] - self.last_action
        self.last_action = self.sim[self.action_props[0]]

    def initialize_custom_properties(self):
        """ Initializes all the custom properties to start values

        Resets the setpints to the initial values

        TODO: check if this can integrated with update_custom_properties
        """
        #reset setpoints to initial values:
        for sp, val in zip(self.setpoint_value_props, self.initial_setpoint_values):
            self.sim[sp] = val
        #now set the custom_props to the start-of-episode values
        error = self.sim[self.setpoint_props[0]] - self.sim[self.setpoint_value_props[0]]    #only one setpoint in SingleChannel_FlightAgentTask
        if self.measurement_in_degrees:
            error = reduce_reflex_angle_deg(error)
        self.sim[self.prop_error] = error
        self.sim[self.prop_error_integral] = error * self.dt    #TODO: is it correct to add the initial error to the integral or shold it be added just _after_ the next timestep
        self.sim[self.prop_delta_cmd] = 0
        self.last_action = self.sim[self.actuating_prop]

    def get_props_to_output(self) -> List[prp.Property]:
        output_props = self.custom_props + [self.setpoint_value_props[0]]
        return output_props

    def change_setpoints(self, new_setpoints: Dict[BoundedProperty, float]):
        """
        Changes the setpoints for the AgentTask. The changes will take effect within the next environment step. (call to env.step())
        The setpoint values are stored within a property in the env's sim object.

        If needed, clean-up actions shall be performed here (like e. g. reset integrators)

        :param new_setpoints: A dictionary with new setpoints to be used. New values overwrite old ones.
        """
        for prop, value in new_setpoints.items():
            try:
                idx = self.setpoint_props.index(prop)
                self.sim[self.setpoint_value_props[idx]] = value    #update the setpoints in the sim_object
                self.sim[self.prop_error_integral] = 0              #reset the integal of the error
            except ValueError:
                #ok, it's not in the list, so it's not for me and I can ignore it
                pass

import numpy as np
import os
import inspect
import gym
import random
import types
from typing import List, Dict, Tuple, Callable, Optional

from collections import namedtuple
from abc import ABC, abstractmethod

import markov_pilot.environment.properties as prp
import markov_pilot.environment.rewards as rewards
from markov_pilot.environment.rewards import RewardComponent
from markov_pilot.environment.assessors import AssessorImpl
from markov_pilot.environment.properties import BoundedProperty
from markov_pilot.environment.simulation import Simulation
from markov_pilot.environment.environment_eee import JsbSimEnv_multi_agent

from markov_pilot.helper.utils import reduce_reflex_angle_deg


class AgentTask(ABC):
    """
    Interface for AgentTasks, which implement the reward calculation and state observation in a multi-agent environment.

    An AgentTask defines its observation space, custom properties, setpoints, action space, termination conditions and agent_reward function.

    A List of AgentTasks is injected into a Multi-Agent environment to handle the observations, the rewards and the done flags.
    """
    def __init__(self, name: str, 
                 make_base_reward_components: Callable[['AgentTask'], Tuple[RewardComponent, ...]] = None,
                 is_done: Callable[['AgentTask'], bool] = None):
        """ Each Agent Taskt needs those porperty lists initialized to be queried.

        The content of the lists must be set for each AgentTask individually.
        Each list entry shall be of type BoundedProperty (TODO: Do I need to define all of them in the properties.py file, or can there be local ones as well?)
        """

        #save the call parameters to init_dict for lab journal
        self.init_dict = {
            'name': name, 
        }


        self.env = None     #we need a reference to the env later on. Shall be injected exactly once by a call to inject_environment() during environment setup
        self.sim = None     #we need a reference to the sim later on. Shall be injected exactly once by a call to inject_environment() during environment setup

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


        return AssessorImpl(base_components, (), positive_rewards=self.positive_rewards)
    
    def inject_environment(self, env: JsbSimEnv_multi_agent):
        """ Injects the environment, the AgentTask is acting in.
        Mostly used to have access to the env.sim object for data storage and retrieval.
        
        It's really easier to consider the AgentTasks as part of the environment with access to the sim object.
        """
        if self.env:
            raise ReferenceError('The inject_environment must be called exactly once per AgentTask. Looks like you called it twice.')
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
    
    @property   #read only, so no setter is specified
    def state_space(self) -> gym.Space:
        """ Get the task's state/observation space object. 
        
        Returns the observation space, the AgentTask operates on.

        Naming is in line with Go-Ren's JsbSimEnv.
        """
        state_lows = np.array([state_var.min for state_var in self.obs_props])
        state_highs = np.array([state_var.max for state_var in self.obs_props])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    @property   #read only, so no setter is specified
    def action_space(self) -> gym.Space:
        """ Get the task's action Space object (maybe an 'empty' Box()"""
        action_lows = np.array([act_prop.min for act_prop in self.action_props])
        action_highs = np.array([act_prop.max for act_prop in self.action_props])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')
    
    @abstractmethod
    def change_setpoints(self, new_setpoints: Dict[BoundedProperty, float]):
        """
        Changes the setpoints for the AgentTask. The changes will take effect within the next environment step. (call to env.step())
        The setpoint values are stored within a property in the env's sim object.

        If needed, clean-up actions shall be performed here (like e. g. reset integrators which is most likely undesired see commment in SingleChannel_FlightAgentTask)

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
        print(f"Observation Space:\n", self.state_space, "\nlow:  ", self.state_space.low, "\nhigh: ", self.state_space.high, sep="")

        custom_props_list = [prop.name for prop in self.custom_props]
        print(f"custom_props[{len(self.custom_props)}]:", end="")
        print(*custom_props_list, sep = ", ")

        action_props_list = [prop.name for prop in self.action_props]
        print(f"action_props[{len(self.action_props)}]:", end="")
        print(*action_props_list, sep = ", ")
        print(f"Action Space: ", self.action_space, "\nlow:  ", self.action_space.low, "\nhigh: ", self.action_space.high, sep="")

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

    def save_make_base_reward_components(self, basedir):
        """
        saves the source code of the passed make_base_reward_components function to a file in the basedir
        
        :param basedir: the directory to save the source code file to
        """
        filename = os.path.join(basedir, f'{self.name}_make_base_reward_components.py')
        os.makedirs(basedir, exist_ok=True)
        with open(filename, 'w') as file:  
            file.write('import markov_pilot.environment.rewards as rewards\n')
            file.write('import markov_pilot.environment.properties as prp\n')
            file.write('from typing import List, Dict, Tuple, Callable, Optional\n')
            file.write('\n')
            file.write(inspect.getsource(self._make_base_reward_components))

        return {
            'make_base_reward_components_file': f'{self.name}_make_base_reward_components.py',
            'make_base_reward_components_fn': self._make_base_reward_components.__name__,
        }


class SingleChannel_FlightAgentTask(AgentTask): #TODO: check whether it would be better to call it SingleAngularChannel_FlightAgentTask(AgentTask)
    """ A class to implement a controller for a single channel actuation with a single error measurement.

    The class SingleChannel_FlightAgentTask takes the value to be controlled as an input state. 

    The SingleChannel_FlightAgentTask calculates the error wrt. the setpoint and some limited error integral to the custom properties. 
    Additionally, the current action and the last actuator travel (Δ-Value) are added to the custom properties and the observation.

    The reward uses the approach of Gor-Ren. The make_base_reward_components-Function shall be injected to the 
    AgentTask object at instantiation time.
    """

    def __init__(self, name: str, actuating_prop: BoundedProperty = None,  
                setpoints: Dict[BoundedProperty, float] = [], #TODO: either make this a single value or make measurement_in_degrees, integral_limit and max_allowed error lists as well.
                presented_state: List[BoundedProperty] = [], 
                make_base_reward_components: Callable[['AgentTask'], Tuple[RewardComponent, ...]] = None,
                is_done: Callable[['AgentTask'], bool] = None, 
                # change_setpoint_callback: Callable[[float], None] = None, #TODO: we used to have that, but it's not used anymore, remove
                measurement_in_degrees = True, max_allowed_error = None,
                integral_limit = float('inf'), integral_decay = 1):
        """
        :param actuating_prop: The actuation variable to be controlled
        :param setpoints: The setpoint property to be used for deviation calculation. On the setpoint properties, the derivative and the integral are calculated and presented in the state.
        :param presented_state: The additional state properties that shall be presented to the Agent besides the props defined within the AgentTask. 
        :param make_base_reward_components = None: Inject a custom function to be bound to instance.
        :param is_done = None: Inject a custom function to be bound to instance.
            Default just checks for max_allowed_error in the self.prop_error custom property.
        :param measurement_in_degrees: indicates if the controlled property and the setpoint is given in degrees
        :param max_allowed_error = 30: The maximum absolute error, before an episode ends. Be careful with setpoint changes! Can be set to None to disable checking.
        :param integral_limit = float('inf'): the limiting value for the error integrator Error integral is clipped to [-integral_limit, integral_limit]
        :param integral_decay = 1: the decay factor for the integral value
        """
        super(SingleChannel_FlightAgentTask, self).__init__(name, 
                                            make_base_reward_components=make_base_reward_components,
                                            is_done=is_done)

        self.init_dict.update({
            'actuating_prop': actuating_prop,  
            'presented_state': presented_state,   
            'measurement_in_degrees': measurement_in_degrees, 
            'max_allowed_error': max_allowed_error,
            'integral_limit': integral_limit, 
            'integral_decay': integral_decay
        })

        if setpoints:
            self.init_dict.update({
                'setpoint_props': list(setpoints.keys()), 
                'setpoint_values': list(setpoints.values()), 
            })

        self.measurement_in_degrees = measurement_in_degrees
        self.max_allowed_error = max_allowed_error
        self.integral_limit = integral_limit
        self.integral_decay = integral_decay
        self.presented_state = presented_state
        self.actuating_prop = actuating_prop

        #TODO: is it really necessary to have self.actuating_prop and self.action_props separated?
        self.action_props = [actuating_prop] if actuating_prop else []

        # the value of the setpoint_prop itself is not really relevant, it must be in the simulator object to be retrieved, 
        # but no need to have it in the obs_props. It's only relevant to be retrieved in the error calculation in update_custom_properties() 
        if setpoints:
            self.setpoint_props, self.initial_setpoint_values = zip(*setpoints.items())    #returns immutable tuples
            self.setpoint_value_props = [sp.prefixed('setpoint') for sp in self.setpoint_props]
        else:
            self.setpoint_props = []
            self.initial_setpoint_values = []
            self.setpoint_value_props = []

        self.define_custom_properties()
        self.define_obs_props()

        self.assessor = self._make_assessor()   #this can only be called after the preparation of all necessary props
        self.print_info()

    def define_custom_properties(self):
        """
        defines the custom properties that are calculated by the SingleChannel_FlightAgentTask
        """
        #custom properties 
        if self.setpoint_props: #there may be tasks without a setpoint
            self.prop_error = BoundedProperty('error/'+self.name+'_err', 'error to desired setpoint', -float('inf'), float('inf'))
            self.prop_error_derivative = BoundedProperty('error/'+self.name+'_deriv', 'derivative of the error divided by timestep', -float('inf'), +float('inf'))
            self.prop_error_integral = BoundedProperty('error/'+self.name+'_int', 'integral of the error multiplied by timestep', -float('inf'), +float('inf'))
        
        if self.actuating_prop: 
            self.prop_delta_cmd = BoundedProperty('info/'+self.name+'_delta-cmd', 'the actuator travel/movement since the last step', 
                self.actuating_prop.min - self.actuating_prop.max, self.actuating_prop.max - self.actuating_prop.min)

        self.custom_props = []
        if self.setpoint_props: self.custom_props.extend([self.prop_error, self.prop_error_derivative, self.prop_error_integral])
        if self.action_props:  self.custom_props.extend([self.prop_delta_cmd])

    def define_obs_props(self):
        """
        Defines the obs_props presented to the agent

        Additionally, the props, which are evaluated in the reward components will automatically be added
        """
        # no need to add the self.prop_delta_cmd to the standard set of obs_props as it will be added if it is evaluated in the reward function
        # self.obs_props = [self.prop_error, self.prop_error_derivative, self.prop_error_integral, self.prop_delta_cmd] + self.presented_state
        self.obs_props = []
        if self.setpoint_props:
            self.obs_props +=[self.prop_error, self.prop_error_derivative, self.prop_error_integral]
        self.obs_props += self.presented_state

    def _make_base_reward_components(self):     #may be overwritten by injected custom function
        # pylint: disable=method-hidden
        """
        Just adds an Asymptotic error component as standard reward to the PID_AgentTask.

        May be overwritten by injected custom function.
        """
        base_components = (     
            rewards.ConstantDummyRewardComponent(name = self.name+'_dummyRwd', const_output=0.0),
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
    
    def update_custom_properties(self): #TODO: make all the calculations suitable for lists; it's easy
        if self.setpoint_props:     #there may be tasks without a setpoint
            cur_value = self.sim[self.setpoint_props[0]]    #only one setpoint for SingleChannel_FlightAgentTask
            error = cur_value - self.sim[self.setpoint_value_props[0]]   #only one setpoint for SingleChannel_FlightAgentTask
            if self.measurement_in_degrees:
                error = reduce_reflex_angle_deg(error)
            self.sim[self.prop_error] = error

            new_derivative = (cur_value - self._last_value) / self.dt
            self.sim[self.prop_error_derivative] = new_derivative

            self.sim[self.prop_error_integral] = np.clip(    #clip the maximum amount of the integral
                            self.sim[self.prop_error_integral] * self.integral_decay + error * self.dt,
                            -self.integral_limit, self.integral_limit
                        )
            self._last_value = cur_value

        if self.action_props != []: #there may be tasks without an associated action
            self.sim[self.prop_delta_cmd] = self.sim[self.action_props[0]] - self._last_action
            self._last_action = self.sim[self.action_props[0]]

    def initialize_custom_properties(self):
        """ Initializes all the custom properties to start values

        TODO: check if this can integrated with update_custom_properties
        """
        #now set the custom_props to the start-of-episode values
        if self.setpoint_props:     #there may be tasks without a setpoint
            cur_value = self.sim[self.setpoint_props[0]]    #only one setpoint for SingleChannel_FlightAgentTask
            error = cur_value - self.sim[self.setpoint_value_props[0]]   #only one setpoint for SingleChannel_FlightAgentTask
            if self.measurement_in_degrees:
                error = reduce_reflex_angle_deg(error)
            self.sim[self.prop_error] = error

            new_derivative = 0
            self.sim[self.prop_error_derivative] = new_derivative

            self.sim[self.prop_error_integral] = np.clip(    #clip the maximum amount of the integral
                            error * self.dt,
                            -self.integral_limit, self.integral_limit
                        )
            self._last_value = cur_value

        if self.action_props != []: #there may be tasks without an associated action
            self.sim[self.prop_delta_cmd] = 0
            if self.actuating_prop:
                self._last_action = self.sim[self.actuating_prop]

    def get_props_to_output(self) -> List[prp.Property]:
        output_props = self.custom_props
        if self.setpoint_value_props: output_props.extend([self.setpoint_value_props[0]])
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
                # TODO: don't reset the error integral as this would be contraproductive in hierarchical \
                # settings where the setpoints are changed by a superodinate controller which will most likely apply only slight changes in each step. \
                # With reset of the integral, this would most likely lead to undesired behavior. This integrator wind-up should be \
                # mitigated by limiting the integral value to suitable bounds (see also https://en.wikipedia.org/wiki/Integral_windup)
                # self.sim[self.prop_error_integral] = 0              #reset the integral of the error
            except ValueError:
                #ok, it's not in the list, so it's not for me and I can ignore it
                pass


class SingleChannel_MinimumProps_Task(SingleChannel_FlightAgentTask):
    '''The presented obs_props are reduced to the error. No integral  nor derivative is included in the obs_props.
    Everything else is left unchanged.

    Additional properties can be included by adding them to the presented_state list
    '''

    def define_obs_props(self):
        """
        Defines the obs_props presented to the agent

        Additionally, the props, which are evaluated in the reward components will automatically be added
        """
        # no need to add the self.prop_delta_cmd to the standard set of obs_props as it will be added if it is evaluated in the reward function
        # self.obs_props = [self.prop_error, self.prop_error_derivative, self.prop_error_integral, self.prop_delta_cmd] + self.presented_state
        self.obs_props = []
        if self.setpoint_props:
            self.obs_props +=[self.prop_error]
        self.obs_props += self.presented_state

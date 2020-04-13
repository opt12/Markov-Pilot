import gym
import numpy as np
import math
from collections import namedtuple

import types
from typing import Type, Tuple, Dict, List, Sequence, NamedTuple, Optional

from gym_jsbsim.simulation import Simulation
from gym_jsbsim.visualiser import FigureVisualiser, FlightGearVisualiser, TimeLineVisualiser
from gym_jsbsim.aircraft import Aircraft, cessna172P
import gym_jsbsim.properties as prp
from gym_jsbsim.properties import BoundedProperty, Property


class JsbSimEnv_multi_agent(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL multi-agent environment conforming to the OpenAI Gym Env
    interface. Used to train multiple agents in a Markov-Game like described in 
    
      - Markov Game: https://www2.cs.duke.edu/courses/spring07/cps296.3/littman94markov.pdf
      - Multi-Agent Env: https://arxiv.org/pdf/1706.02275.pdf
      - Code for Multi-Agent Env: https://github.com/openai/multiagent-particle-envs

    Instead of a single agent, multiple agents act on the same environment and
    receive 
      - individual observations (obs ⊆ state)
      - individual rewards
      - individual done flags

    An JsbSimEnv_multi_agent is instantiated with a list of Task_multi_agent objects.
    Each implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.

    ATTRIBUTION: this class implements the OpenAI Gym Env API. Method
    docstrings have been adapted or copied from the OpenAI Gym source code.
    """
    JSBSIM_DT_HZ: int = 60  # JSBSim integration frequency
    INITIAL_ALTITUDE_FT_default = 6000
    INITIAL_GEO_POSITION_LAT_LON_default = (53.248488, 10.459216)   #coordinates of the Lüneburg Airfield EDHG
    THROTTLE_CMD_default = 0.8
    MIXTURE_CMD_default = 0.8
    INITIAL_HEADING_DEG_default = 270
    DEFAULT_EPISODE_TIME_S = 120.

    metadata = {'render.modes': ['human', 'flightgear', 'timeline']}
    state_props: Tuple[Property, ...]
    # action_variables: Tuple[Property, ...]
    State: Type[namedtuple]
    Actions: Type[namedtuple]

    def __init__(self, task_list, aircraft: Aircraft = cessna172P,
                 agent_interaction_freq: int = 5, episode_time_s: float = DEFAULT_EPISODE_TIME_S):
        """
        Constructor. Inits some internal state, and calls JsbSimEnv_multi_agent.reset()
        for a first time to prepare the internal simulation sim object.

        :param task_list: the list of Task_multi_agent instances to take part in the Markov Game
        :param aircraft: the JSBSim aircraft to be used
        :param agent_interaction_freq: int, how many times per second the agent
            should interact with environment.
        """
        if agent_interaction_freq > self.JSBSIM_DT_HZ:
            raise ValueError('agent interaction frequency must be less than '
                             'or equal to JSBSim integration frequency of '
                             f'{self.JSBSIM_DT_HZ} Hz.')
        self.sim_steps_per_agent_step: int = self.JSBSIM_DT_HZ // agent_interaction_freq
        self.dt = 1/agent_interaction_freq  #length of a timestep in seconds        
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * agent_interaction_freq)
        self.steps_left = BoundedProperty('info/steps_left', 'steps remaining in episode', 0,
                                          episode_steps)

        self.engines_running = False    #we usually use a gliding descent
        self.throttle_cmd, self.mixture_cmd = self.THROTTLE_CMD_default, self.MIXTURE_CMD_default

        self.aircraft = aircraft
        self.task_list = task_list

        self.inital_attitude: Dict[Property, float] = {     #the default initial conditions; shall be overwritten by calling set_initial_conditions()
              prp.initial_u_fps: self.aircraft.get_cruise_speed_fps()*0.9    #forward speed
            , prp.initial_v_fps: 0   # side component of speed; shall be 0 in steady flight
            , prp.initial_w_fps: 0   # down component of speed; shall be 0 in steady flight
            , prp.initial_p_radps: 0 # angular velocity roll
            , prp.initial_q_radps: 0 # angular velocity pitch
            , prp.initial_r_radps: 0 # angular velocity yaw

            , prp.initial_flight_path_deg: 0
            , prp.initial_roll_deg:0
            , prp.initial_aoa_deg: 1.0    #just an arbitrary value for a reasonable AoA

            , prp.initial_altitude_ft: self.INITIAL_ALTITUDE_FT_default
            , prp.initial_terrain_altitude_ft: 0.00000001
            , prp.initial_latitude_geod_deg: self.INITIAL_GEO_POSITION_LAT_LON_default[0]
            , prp.initial_longitude_geoc_deg: self.INITIAL_GEO_POSITION_LAT_LON_default[1]
            #TODO: check whether some of those values could go into basic_ic.xml or minimal_ic.xml
        }

        #initialize a Simulation object to have access to the storage; 
        self.sim: Simulation = self._init_new_sim(self.JSBSIM_DT_HZ, self.aircraft, initial_conditions = self.inital_attitude)
        [t.inject_environment(env=self) for t in task_list]    #inject a reference to the env into the task

        obs_props: prp.BoundedProperty = []
        self.custom_props: prp.BoundedProperty = [self.steps_left]
        self.setpoint_props: prp.BoundedProperty = []
        self.action_props: prp.BoundedProperty = []
        self.observation_spaces: List[gym.spaces.Box] = []
        self.action_spaces: List[gym.spaces.Box] = []
        for t in self.task_list:
            obs_props.extend(t.obs_props)
            self.custom_props.extend(t.custom_props)
            self.setpoint_props.extend(t.get_setpoint_props())
            self.action_props.extend(t.action_props)
        
        unique_setpoint_props = { sp_prop for sp_prop in self.setpoint_props }  #this is a set comprehension to convert the list of props into a set with unique entries
        if len(self.setpoint_props) != len(unique_setpoint_props):
            #the setpoint properties are not unique in between the tasks
            raise ValueError('The setpoint properties in the different tasks must be unique\n')

        unique_custom_props = { c_prop for c_prop in self.custom_props }  #this is a set comprehension to convert the list of props into a set with unique entries
        if len(self.custom_props) != len(unique_custom_props):
            #the custom properties are not unique in between the tasks
            raise ValueError('The custom properties in the different tasks must be unique\n')

        state_props = obs_props
        state_props.extend(self.custom_props) #in the overall-state there must be everything that is observed or manipulated by any task
        self.state_props = list({ st_prop for st_prop in state_props }) #this is a set comprehension to convert the list of props into a list from a set with unique entries
        self.state_names = [prop.get_legal_name() for prop in self.state_props]
        self.State = namedtuple('State', self.state_names)

        unique_action_props = { act for act in self.action_props }  #this is a set comprehension to convert the list of props into a set with unique entries

        if len(self.action_props) != len(unique_action_props):
            #the actions are not unique in between the tasks
            raise ValueError('The actions in the different tasks must be unique\n')
        
        self.action_names = [prop.get_legal_name() for prop in self.action_props]
        self.Actions = namedtuple('Actions', self.action_names)

        
        self.obs_idxs = []
        # self.act_idxs = []
        for t in self.task_list:
            #get the indexes 
            obs_prop_names = [prop.get_legal_name() for prop in t.obs_props]
            task_obs_idxs = [self.state_names.index(name) for name in obs_prop_names]
            self.obs_idxs.append(task_obs_idxs)
            act_prop_names = [prop.get_legal_name() for prop in t.action_props]
            # task_act_idxs = [self.action_names.index(name) for name in act_prop_names]
            # self.act_idxs.append(task_act_idxs)
            # set Space objects
            lower_obs_limits = np.array([self.state_props[idx].min for idx in task_obs_idxs], dtype=np.float32)
            upper_obs_limits = np.array([self.state_props[idx].max for idx in task_obs_idxs], dtype=np.float32)
            task_observation_space = gym.spaces.Box(lower_obs_limits, upper_obs_limits, dtype=np.float32)
            self.observation_spaces.append(task_observation_space)
            lower_act_limits = np.array([av.min for av in t.action_props], dtype=np.float32)
            upper_act_limits = np.array([av.max for av in t.action_props], dtype=np.float32)
            task_action_space = gym.spaces.Box(lower_act_limits, upper_act_limits, dtype=np.float32)
            self.action_spaces.append(task_action_space)

        # set visualisation objects
        self.flightgear_visualiser: FlightGearVisualiser = None
        self.meta_dict = {'model_type': 'trained', 'model_discriminator': 'no file'} #to store additional meta information (e. g. used ny Episode Plotter Wrapper)

    def reset(self) -> List[np.ndarray]:
        """
        Resets the state of the environment and returns an initial observation.

        :return: array, the initial observation of the space.
        """
        self.sim.reinitialise(self.inital_attitude)
        self.state = self._observe_first_state(self.sim) #TODO: adapt this!
        self.last_state = self.state #set the last state the same as the first state at start of each episode

        if self.flightgear_visualiser:
            self.flightgear_visualiser.configure_simulation_output(self.sim)

        self.obs_n = self._get_obs_from_state(self.state)
        self.last_obs_n = self.obs_n #set the last observation the same as the first observation at start of each episode

        return self.obs_n

    def _observe_first_state(self, sim: Simulation) -> np.ndarray:
        self._new_episode_init()
        self._initialize_custom_properties()
        state: NamedTuple(float) = self.State(*(sim[prop] for prop in self.state_props))
        return state

    def _initialize_custom_properties(self):
        """
        Iitializes any custom properties.

        First, the environmental custom properties are initialized (empty at the moment)
        and later on, all TaskAgent specific custom properties are initialized.

        TODO: check if this can integrated with _update_custom_properties()
        """
        #update environments custom_props
        self.sim[self.steps_left] = self.steps_left.max

        #update Task_Agent specific custom_props
        [t.initialize_custom_properties() for t in self.task_list]

    def _update_custom_properties(self):
        """
        Updates any custom properties.

        First, the environmental custom properties are updated (empty at the moment)
        and later on, all TaskAgent specific custom properties are updated.
        """
        #update environments custom_props
        self.sim[self.steps_left] -= 1  #decrement the remaining task steps
    
    def is_terminal(self):
         return not (self.sim[self.steps_left] > 0)

    def _get_obs_from_state(self, state_tuple) -> List[np.ndarray]:
        """ creates the individual obserations for each Agent_Task
        """
        state_array = np.array(state_tuple)
        obs_n = [state_array.take(self.obs_idxs[i]) for i in range(len(self.task_list))]
        return obs_n

    def _new_episode_init(self) -> None:
        """
        This method is called at the start of every episode. It is used to set
        the value of any controls or environment properties not already defined
        in the task's initial conditions.

        By default it 
          - raises the landing gear 
          - optionally starts the engines
          - resets the episode step counter
        """
        self.sim.raise_landing_gear()
        if self.engines_running:
            self.sim.start_engines()   #TODO:   we need some control to configure the engines, but for the moment we go into gliding descent
            self.sim.set_throttle_mixture_controls(self.throttle_cmd, self.mixture_cmd)        

    def step(self, actions_n: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict]]:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts a list of actions and returns a tuple (list of observations, list of rewards, list of dones, list of info-dicts).
        I. e. each agent issues the same data like in a single agent environment, but everything has to be concatenated to a list before.
        All results are also returned as a list with one entry for each agent which has to be destructured properly.

        :param actions: the list of the agents' actions.
        :return:
            obs_n: list of the agents' observations like specified in the task
            reward_n: list of the agents' rewards like specified in the task
            done_n: list whether the episode has ended for a certain task, in which case further step() calls are undefined
            info_n: auxiliary information from each task, e.g. a dictinary of all reward components
        """
        if len(actions_n) != len(self.task_list):
            raise ValueError('mismatch between action list and task list length')

        self.last_state = self.state                 #stash the last_state
        self.last_obs_n = self.obs_n                 #stash the last observations
        self.state = self._issue_actions(actions_n)  #issue actions to the simulator to update the state

        #calculate rewards for each task and check for dones, gather extra info contents
        self.obs_n = self._get_obs_from_state(self.state) #update the individual observations for each task
        rwd_n, done_n, info_n = zip(*[t.assess(self.obs_n[i], self.last_obs_n[i]) for i, t in enumerate(self.task_list)])
        
        #return the result tuple (obs_n, rwd_n, done_n, info_n)
        return self.obs_n, rwd_n, done_n, info_n
            
    def _issue_actions(self, actions_n: List[np.ndarray]) -> Tuple[NamedTuple]:
        """
        Perform a simulation step in the environment and return the new state

        Depending on the interaction frequency, the simulator runs more than one step.

        :param actions: the list of actions to issue to the simulator
        :return: a named Tuple with the new State 
        """
        #flatten the actions
        actions = []
        [actions.extend(act) for act in actions_n]
        if len(actions) != len(self.action_props):
            raise ValueError('mismatch between the total number of actions and actions registered by the individual tasks')
        actions = np.array(actions, dtype=np.float32)   # we need it as a numpy array

        # input actions
        '''in the following line it is important, that the action_props are ordered the same way, 
           the actions are ordered in the call to step. that's why we must not change the order of 
           actions in the __init__() constructor.
        '''
        for prop, act in zip(self.action_props, actions):
            self.sim[prop] = act

        # run simulation
        for _ in range(self.sim_steps_per_agent_step):
            if(not self.sim.run()):   #TODO: check return value. is it false if JSBSim encounters a problem
                raise RuntimeError("JSBSim terminated")
        
        #update the custom propeties of the env
        self._update_custom_properties()
        #update Task_Agent specific custom_props
        [t.update_custom_properties() for t in self.task_list]
        
        state_new: NamedTuple(float) = self.State(*(self.sim[prop] for prop in self.state_props))   # enter the values to a variable of the State class (named tuple)
        # if self.debug:
        #     self._validate_state(state_new)   #returns true if state contains nan

        return state_new

    def _init_new_sim(self, dt, aircraft, initial_conditions) -> Simulation:
        return Simulation(sim_frequency_hz=dt,
                          aircraft=aircraft,
                          init_conditions=initial_conditions,
                          allow_flightgear_output=True)
    # @timeit
    def render(self, mode='flightgear', flightgear_blocking=True):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        :param mode: str, the mode to render with
        :param flightgear_blocking: waits for FlightGear to load before
            returning if True, else returns immediately
        """
        if mode == 'human':
            if not self.figure_visualiser:
                self.figure_visualiser = FigureVisualiser(self.sim,
                                                          self.get_props_to_output())
            self.figure_visualiser.plot(self.sim)
        elif mode == 'timeline':
            if not self.timeline_visualiser:
                self.timeline_visualiser = TimeLineVisualiser(self.sim,
                                                          self.get_timeline_props_to_output())
            self.timeline_visualiser.plot(self.sim)
        elif mode == 'flightgear':
            if not self.flightgear_visualiser:
                self.flightgear_visualiser = FlightGearVisualiser(self.sim,
                                                                  flightgear_blocking)
            self.flightgear_visualiser.plot(self.sim)
        else:
            super().render(mode=mode)

    def close(self):
        """ Cleans up this environment's objects

        Environments automatically close() when garbage collected or when the
        program exits.
        """
        if self.sim:
            self.sim.close()
        if self.figure_visualiser:
            self.figure_visualiser.close()
        if self.timeline_visualiser:
            self.timeline_visualiser.close()
        if self.flightgear_visualiser:
            self.flightgear_visualiser.close()

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        gym.logger.warn("Could not seed environment %s", self)
        return

    def set_initial_conditions(self, initial_conditions: Optional[Dict[Property, float]] = None):
        """
        Sets the initial conditions for the episodes withihn JSBSim.

        Episode initial conditions (ICs) are defined by specifying values for
        JSBSim properties, represented by their name (string) in JSBSim.

        JSBSim uses a distinct set of properties for ICs, beginning with 'ic/'
        which differ from property names during the simulation, e.g. "ic/u-fps"
        instead of "velocities/u-fps". See https://jsbsim-team.github.io/jsbsim/

        :param initial_conditions: dict mapping string for each initial condition property to
            initial value, a float. Only given conditions are overwritten in the sim object.
            There is no possibility (and no necessity) to delete certain conditions.
        """
        if initial_conditions is not None:
            for prop, value in initial_conditions.items():
                self.inital_attitude[prop] = value  #update the initial conditions

    def get_state_property_names(self):
        """
        returns the list of names of the named tuple representing the state of the specific task
        """
        return self.task.State._fields
    
    def set_meta_information(self, **kwargs):
        self.meta_dict = {**self.meta_dict, **kwargs}

    def change_setpoints(self, setpoints: Dict[Property, float]):
        """
        Forwards the new setpoints to all AgentTasks. They decide on their own whether it matters for them.

        (Why not just update the setpoints in the sim object? -> The AgentTasks read it from there, as we want to reset the integrators inside the AgentTasks)
        """
        [t.change_setpoints(setpoints) for t in self.task_list]

# class NoFGJsbSimEnv_multi_agent(JsbSimEnv_multi_agent):
#     """
#     An RL environment for JSBSim with rendering to FlightGear disabled.

#     This class exists to be used for training agents where visualisation is not
#     required. Otherwise, restrictions in JSBSim output initialisation cause it
#     to open a new socket for every single episode, eventually leading to
#     failure of the network.
#     """
#     metadata = {'render.modes': ['human']}

#     def _init_new_sim(self, dt: float, aircraft: Aircraft, initial_conditions: Dict):
#         return Simulation(sim_frequency_hz=dt,
#                           aircraft=aircraft,
#                           init_conditions=initial_conditions,
#                           allow_flightgear_output=False)

#     def render(self, mode='human', flightgear_blocking=True):
#         if mode == 'flightgear':
#             raise ValueError('flightgear rendering is disabled for this class')
#         else:
#             super().render(mode, flightgear_blocking)

#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import gym
import os
import pickle
import json
from gym.spaces import Box
import numpy as np

import types
from typing import Type, Tuple, Dict, List, Sequence, NamedTuple, Optional, Union

from collections import namedtuple

from markov_pilot.helper.utils import aggregate_gym_boxes
from markov_pilot.helper.bunch import Bunch
from markov_pilot.agents.AgentTrainer import Experience, AgentTrainer, PID_AgentTrainer, DDPG_AgentTrainer, MADDPG_AgentTrainer

AgentSpec = namedtuple('AgentSpec', ['name', 'agent_type', 'task_names', 'parameters'])

class AgentContainer():
    """
    """

    def __init__(self, task_list_n: List['AgentTask'], agents_m: List['AgentTrainer'], mapping_dict: Dict[str, List[str]]):
        """
        
        :param task_list_n: the list with references off all tasks in the environemnt; used to query structural information
        :param agents_m: the list of agents; to be maintained by AgentContainer;
        :param mapping_dict: A dictionary mapping one agent.name to  1..n [task.name]; used to distribute inputs and actions
        """
        #extract the needed information form the parameters
        reduced_task_list_n = [Bunch(name= t.name, action_space = t.action_space, state_space = t.state_space) \
                    for t in task_list_n]    #we need to strip off unnecessary information to serialize the container properly later on

        self.task_list_n = reduced_task_list_n
        self.agents_m = agents_m
        self.m = len(self.agents_m)    #the number of Agents registered with the agent container
        self.mapping_dict = mapping_dict
        
        self.init_dict = {
            'task_list_n': reduced_task_list_n,
            'agents_m': self.agents_m,
            'mapping_dict': mapping_dict,
        }
        
        #to restore the agents, we need their classes
        self.agent_classes_dict = {ag.type: ag.__class__ for ag in self.agents_m}
        self.agent_init_dict_m = [ag.agent_dict for ag in self.agents_m]

        self.task_names = list(map(lambda t: t.name, self.task_list_n))
        #build a list containing the list of associated task idxs for each agent idx
        self.task_idxs_per_agent: List[List[int]] = []
        self.agent_idx_per_task = np.full( shape=len(self.task_list_n), fill_value=-1, dtype=np.int32) 
        for i, ag in enumerate(self.agents_m):
            t_idxs = [self.task_names.index(n) for n in self.mapping_dict[ag.name]]
            self.task_idxs_per_agent.append(t_idxs)
            if any([True for ag_idx in self.agent_idx_per_task[t_idxs] if ag_idx != -1]):
            #we had a double entry
                raise ValueError("All tasks in task_list_n must be contained in the mapping_dict at most once.")
            np.put(self.agent_idx_per_task, t_idxs, i)


        # build the action space for each agent
        self.task_action_space_n = list(map(lambda t: t.action_space, self.task_list_n))
        self.agent_action_space_n = []
        self.task_to_agent_action_idxs = [None] * len(self.task_list_n)
        #for each agent
        act_start_idx = 0   #the index of the first action in the aggregated actions from all agents
        for a_idx in range(len(self.agents_m)):
            ag_act_space_n = []
            #for each associated task
            for t_idx in self.task_idxs_per_agent[a_idx]:
                ag_act_space_n.append(self.task_action_space_n[t_idx])
                action_width = ag_act_space_n[-1].shape[-1]
                #determine the action idxs for each task in the concatenation of all agents' actions
                self.task_to_agent_action_idxs[t_idx] = list(range(act_start_idx, act_start_idx+action_width))
                act_start_idx += action_width
            # we have the task specific action spaces in ag_act_space_n now
            self.agent_action_space_n.append(aggregate_gym_boxes(ag_act_space_n)) 
            
    def get_action(self, obs_n, add_exploration_noise = False):
        agent_obs_m = self._get_per_agent_data(obs_n)
        agent_actions_m = [ag.get_action(obs, add_exploration_noise) for ag, obs in zip(self.agents_m, agent_obs_m)]
        task_actions_n = self._get_per_task_action(agent_actions_m)
        return task_actions_n
    
    def get_per_agent_experience(self, obs_n, actions_n, rewards_n, next_obs_n, dones_n):
        exper_trans_m = list(map(self._get_per_agent_data, (obs_n, actions_n, rewards_n, next_obs_n, dones_n))) #returns an 5*m object
        # aggregate the rewards
        exper_trans_m[2] = [self.agents_m[i].rwd_aggregator(rwd_list) for i, rwd_list in enumerate(exper_trans_m[2])]
        #aggregate the dones
        exper_trans_m[4] = [any(done_list) for done_list in exper_trans_m[4]]
        experience_m = [Experience(*e) for e in list(zip(*exper_trans_m))]    # first transpose to m*5 object and then put in Experience

        return experience_m
    
    def remember(self, obs_n, actions_n, rewards_n, next_obs_n, dones_n):
        """
        Forwards the experience to the agents for pushing it into their replay buffers.

        TODO: A centralized replay buffer comes into mind. Check implications
        """
        experience_m = self.get_per_agent_experience(obs_n, actions_n, rewards_n, next_obs_n, dones_n)
        [ag.store_experience(experience) for ag, experience in zip(self.agents_m, experience_m)]
        return experience_m

    def train_agents(self):
        """
        Issues a train command to all agents. 
        
        The list of all agents is passed as parameter together with the own idx in this list.

        DDPG agents ignore these parmeters and do pure local training while MADDPG agents can query the replay buffer of the other agents.
        """
        [ag.train(agents_m = self.agents_m, own_idx = own_idx) for own_idx, ag in enumerate(self.agents_m)]

    def _get_per_agent_data(self, task_input_n: List[Union[float, np.ndarray]]) -> List[np.ndarray]:
        """
        Aggregates the inputs from 1..n tasks into the single input for one agent.

        Input is either a list of arrays or a list of values

        :return: a list with the data for each agent taken form the task_data. 
        """
        #concatenate all inputs
        np_inp = np.array(task_input_n)
        # for each agent select the inputs and concatenate
        agent_inputs = [np.hstack(np_inp[self.task_idxs_per_agent[ag_idx]]) for ag_idx in range(len(self.task_idxs_per_agent))]
        return agent_inputs
        
    def _get_per_task_action(self, agent_action_n: List[np.ndarray]) -> List[np.ndarray]:
        """
        Decomposes the actions from 1..m agents into the action-arrays for n tasks

        :return: a list with the action array for each task
        """
        #concatenate all inputs
        np_inp = np.concatenate(agent_action_n)
        # for each task build the list  of actions
        task_actions = [np_inp[self.task_to_agent_action_idxs[t_idx]] if self.task_to_agent_action_idxs[t_idx] else None \
                                        for t_idx in range(len(self.task_list_n)) ]
        return task_actions

    def save_agent_container_data(self, save_path):
        """
        Saves a JSON file with agent container data. Also saves a pickle file
        from which the container can be restored.

        :param save_path: The directory to store the JSON and the pcikle file to
        """

        #the agents themselves are not (JSON-)serializable
        #we only save names and types of agents and load the agents separately when restoring
        data_to_save = {
            'task_list_n_names': [t.name for t in self.task_list_n],
            'agents_m_names': [ag.name for ag in self.agents_m],
            'agents_m_types': [ag.type for ag in self.agents_m],
            'mapping_dict': self.init_dict['mapping_dict'],
        }

        #TODO: add the init dict of each agent to see the network topology
        # this is a bit problematic as Box is not JSON serializable
        # data_to_save.update = {
        #     'agents_init_parameters_m' = [ag.init_params for ag in self.agents_m]
        # }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        #save as JSON as it's readable
        json_filename = os.path.join(save_path, 'agent_container.json')
        with open(json_filename, 'w') as file:
            file.write(json.dumps(data_to_save, indent=4))

        # we need more data to restore the tasks and agents
        data_to_save.update({
            'task_list_n': self.task_list_n,  #task_list_n is reduced in __init__ to be serializable
            'agent_classes_dict': self.agent_classes_dict,
            'agent_init_dict_m': self.agent_init_dict_m
        })

        #this data is also the one to save to the agent_container.pickle
        pickle_filename = os.path.join(save_path, 'agent_container.pickle')
        with open(pickle_filename, 'wb') as file:
            pickle.dump(data_to_save, file)

    @classmethod
    def init_from_save(cls, container_pickle_file: str, agent_pickle_files_m: List[str] = []) ->'AgentContainer':
        """
        restores an agent container from the files on disk. If no agents ar given, new ones are created.
        
        :param container_pickle_file: The file containing the pickle fl efor the container itself
        :param agent_pickle_files_m: A list of agents to be used to inject into the container. Must fit the data in container_pickle_file.
            If left out, new agents are created.
        """
        #load the container pickle
        with open(container_pickle_file, 'rb') as infile:
            container_pickle_data = pickle.load(infile)
        
        #load the agents if given
        if len(agent_pickle_files_m) > 0:
            agent_init_dict_m = []
            for agent_pickle in agent_pickle_files_m:
                with open(agent_pickle, 'rb') as infile:
                    agent_init_dict = pickle.load(infile)
                agent_init_dict_m.append(agent_init_dict)
        else:
            agent_init_dict_m = container_pickle_data['agent_init_dict_m']

        agent_classes_dict = container_pickle_data['agent_classes_dict']

        #instantiate the agents
        agents_m = []
        for agent_init_dict in agent_init_dict_m:
            agents_m.append(agent_classes_dict[agent_init_dict['type']](**agent_init_dict))

        container_dict = {
            'task_list_n': container_pickle_data['task_list_n'], 
            'agents_m': agents_m, 
            'mapping_dict': container_pickle_data['mapping_dict'],
        }

        instance = cls(**container_dict)     #call to the own constructor to initialize the agents
        instance.init_dict = container_dict  #stash away the parameters used for agent instantiation
        return instance

    @classmethod
    def init_from_env(cls, task_list_n: List['Agent_Task'], agent_spec: List[AgentSpec], 
                      agent_classes_dict: Dict['str', 'Class'], **kwargs) -> 'AgentContainer':
        """
        Instantiate the specified agents. Then instantiate an AgentContainer holding those agents

        :param env: The environment whith its tasks listed in env.task_list
        :param agent_spec: a list of agents Specifications. The agent specification is a 
            namedtuple('AgentSpec', ['name', 'type', 'task_names', 'parameters']). 
             - name: the name of the agent
             - type: one of 'PID', 'DDPG' or 'MADDPG'
             - task_names: The task_names associated to the agent
             - parameters: The parameters needed to instantiate the agent. Agent-type dependent.
        :param agent_classes_dict: A dictionary holding a Class for each agent type appearing in the agent_spec.type. 
            Classes shall be subclass of AgentTrainer
        """

        reduced_task_list_n = [Bunch(name= t.name, action_space = t.action_space, state_space = t.state_space) \
                    for t in task_list_n]    #we need to strip off unnecessary information to serialize the container properly later on

        task_names_n = [t.name for t in reduced_task_list_n]
        action_spaces_n = [t.action_space for t in reduced_task_list_n]
        obs_spaces_n = [t.state_space for t in reduced_task_list_n]

        try:
            interaction_frequency = kwargs['interaction_frequency']
        except KeyError:
            print(f'*** INFO: no interaction frequency given for AgentContainer.init_from_env(); Using 5Hz as default.')
            interaction_frequency = 5
        
        # figure out the input and output widths for the agents
        # - actor_obs_space is the concatenation of the associated elements from obs_spaces_n
        # - action_space is the concatenation of the associated elements action_spaces_n
        # for MADDPG agents, an additional critic_state_space needs to be defined which is
        # the concatenation of actor_obs_space + other_agents_obs_space + other_agents_act_space

        #build a list containing the list of associated task idxs for each agent idx
        #determine the observation and action spaces for the actors of each agent
        task_idxs_per_agent: List[List[int]] = []
        ag_actor_obs_spaces_m = []
        ag_actor_act_spaces_m = []
        mapping_dict = {}
        agent_idx_per_task = np.full( shape=len(reduced_task_list_n), fill_value=-1, dtype=np.int32) 
        for i, aspec in enumerate(agent_spec):
            t_idxs = [task_names_n.index(n) for n in aspec.task_names]
            task_idxs_per_agent.append(t_idxs)
            ag_actor_act_spaces_m.append(aggregate_gym_boxes([action_spaces_n[i] for i in t_idxs]))
            ag_actor_obs_spaces_m.append(aggregate_gym_boxes([obs_spaces_n[i] for i in t_idxs]))
            mapping_dict.update({aspec.name:aspec.task_names})
            #check that no doubles appeared
            if any([True for ag_idx in agent_idx_per_task[t_idxs] if ag_idx != -1]):
            #we had a double entry in the task association
                raise ValueError("All tasks in task_list_n must be contained in the mapping_dict at most once.")
            np.put(agent_idx_per_task, t_idxs, i)

        #determine the critic_state_space
        ag_critic_state_space_m = []
        for i, aspec in enumerate(agent_spec):
            if aspec.agent_type != 'MADDPG':
                #this is _not_ an MADDPG and hence, no additional information to the critic
                ag_critic_state_space_m.append(ag_actor_obs_spaces_m[i])
            else:
                #the critic's state is enhanced by the other agent's obs_space and the other agents' act_space
                critic_state_space = [ag_actor_obs_spaces_m[i]]
                critic_state_space.extend([ag_actor_obs_spaces_m[j] for j in range(len(agent_spec)) if j!=i])
                critic_state_space.extend([ag_actor_act_spaces_m[j] for j in range(len(agent_spec)) if j!=i])
                ag_critic_state_space_m.append(aggregate_gym_boxes(critic_state_space))
        
        #now instantiate the agents from the specs
        agents_m = []
        agent_init_dict_m =[]
        for i, aspec in enumerate(agent_spec):
            agent_init_dict = {
                'name': aspec.name,
                'obs_space': ag_actor_obs_spaces_m[i],
                'act_space': ag_actor_act_spaces_m[i],
                'critic_state_space': ag_critic_state_space_m[i],
                'agent_interaction_frequency': interaction_frequency,
            }
            agent_init_dict.update(aspec.parameters)
            agent_init_dict_m.append(agent_init_dict)

            new_agent = agent_classes_dict[aspec.agent_type](**agent_init_dict)
            agents_m.append(new_agent)

        container_dict = {
            'task_list_n': reduced_task_list_n, 
            'agents_m': agents_m, 
            'mapping_dict': mapping_dict
        }

        instance = cls(**container_dict)     #call to the own constructor to initialize the agents
        instance.init_dict = container_dict  #stash away the parameters used for agent instantiation
        return instance


         
if __name__ == '__main__':
    class MiniTask():
        def __init__(self, name, action_space, state_space):
            self.name = name
            self._action_space = action_space
            self._state_space = state_space
        @property
        def action_space(self):
            return self._action_space
        @property
        def state_space(self):
            return self._state_space
    class MiniAgent():
        def __init__(self, name, act_space, **kwargs):
            self.name = name
            self.act_space = act_space
            self.type = self.__class__.__name__
            self.agent_dict = {
                'name': self.name,
                'buf_len': 123,
                'obs_space': Box(np.array([-1, -2]), np.array([1, 2])),
                'act_space': act_space,
                'type': self.type,
                'train_steps': 666,
                'agent_interaction_freq': 5
            }

            print(f'{self.__class__.__name__} instantiated with:')
            for key, value in kwargs.items(): 
                print ("%s = %s" %(key, value)) 


        def get_action(self, obs, add_exploration_noise = False):
            return np.full( shape=self.act_space.shape, fill_value=self.act_space.shape[-1], dtype=np.float32)
        def store_experience(self, experience):
            print(f'{self.name} stored {experience}')
        def rwd_aggregator(self, rwd_list):
            return rwd_list.sum()
    class MiniEnv():
        def __init__(self, task_list):
            self.task_list = task_list

    class PID_Agent(MiniAgent):
        ...
    class DDPG_Agent(MiniAgent):
        ...
    class MADDPG_Agent(MiniAgent):
        ...

    task_list = [
        MiniTask('t1', Box(low=np.array([-1,-2,-3]), high=np.array([1,1,1])), Box(low=np.array([-1]), high=np.array([1]))),
        MiniTask('t2', Box(low=np.array([-1]), high=np.array([1])), Box(low=np.array([-2,-2]), high=np.array([2,2]))),
        MiniTask('t3', Box(low=np.array([-1,-2]), high=np.array([1,12])), Box(low=np.array([-3, -3, -3]), high=np.array([3,3,3]))),
        MiniTask('t4', Box(low=np.array([-1,-10,-12]), high=np.array([1,0,9])), Box(low=np.array([-4,-4,-4,-4]), high=np.array([4,4,4,4]))),
        MiniTask('t5', Box(low=np.array([-1]), high=np.array([1])), Box(low=np.array([-5,-5,-5,-5,-5]), high=np.array([5,5,5,5,5])))
    ]

    agent_spec = [
        AgentSpec('ag1','DDPG', ['t1', 't3'], {'action_width': 4, 'hallo':'hallo', 'DDPG':'DDPG', 'ag_nr':1}),
        AgentSpec('ag3','MADDPG', ['t2', 't4'], {'action_width': 4, 'hallo':'hallo', 'MADDPG':'MADDPG', 'ag_nr':3}),
        AgentSpec('ag2','PID', ['t5'], {'hallo':'hallo', 'PID':'PID', 'ag_nr':2, 'action_width': 4, }),
    ]

    agent_classes_dict = {
        'PID': PID_Agent,
        'MADDPG': MADDPG_Agent,
        'DDPG': DDPG_Agent
    }

    agent_classes_dict['MADDPG']('hello', Box(np.array([-1,-2,-3]), np.array([1,2,3])), hello='world')

    env = MiniEnv(task_list)
    ag_container = AgentContainer.init_from_env(task_list, agent_spec, agent_classes_dict)

    agents_m = [
        MiniAgent('ag1', act_space= Box(np.array([-1,-2,-3]), np.array([1,2,3]))),
        MiniAgent('ag2', act_space= Box(np.array([-1,-2,-3]), np.array([1,2,3]))),
        MiniAgent('ag3', act_space= Box(np.array([-1,-2,-3]), np.array([1,2,3])))
    ] 

    mapping_dict = {
        'ag1': ['t1'],
        'ag3': ['t2'],
        'ag2': ['t5', 't3'],
    }

    a_cont = AgentContainer(task_list, agents_m, mapping_dict)

    a_cont.save_agent_container_data('./test_save/')

    restored_cont = AgentContainer.init_from_save('./test_save/agent_container.pickle')
    restored_cont.save_agent_container_data('./test_save/')

    obs_n = [
        [1],
        [2,2],
        [3,3,3],
        [4,4,4,4],
        [5,5,5,5,5],
    ]

    rwd_n = [[1],[2],[3],[4],[5]]
    rwd_n = [1,2,3,4,5]

    done_n = [[True], [False], [True], [False], [True]]
    done_n = [True, False, True, False, True]

    task_actions_n = a_cont.get_action(obs_n)

    a_cont.remember(obs_n, task_actions_n, rwd_n, obs_n, done_n)

    #now try out with real agents from AgentTrainer class

    from markov_pilot.agents.AgentTrainer import PID_AgentTrainer, DDPG_AgentTrainer, MADDPG_AgentTrainer, PidParameters
    import markov_pilot.environment.properties as prp
    from markov_pilot.tasks.tasks_eee import SingleChannel_FlightAgentTask

    elevator_AT_for_PID = SingleChannel_FlightAgentTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: 66},
                                integral_limit = 100)
                                #integral_limit: self.Ki * dt * int <= output_limit --> int <= 1/0.2*6.5e-2 = 77

    aileron_AT_for_PID = SingleChannel_FlightAgentTask('aileron', prp.aileron_cmd, {prp.roll_deg: 99}, 
                                max_allowed_error= 60, 
                                integral_limit = 100)
                                #integral_limit: self.Ki * dt * int <= output_limit --> int <= 1/0.2*1e-2 = 500

    task_list = [
        elevator_AT_for_PID,
        aileron_AT_for_PID
    ]


    #for PID controllers we need an alaborated parameter set for each type
    pid_params = {'aileron':  PidParameters(3.5e-2,    1e-2,   0.0),
                  'elevator': PidParameters( -5e-2, -6.5e-2, -1e-3)}

    params_aileron_pid_agent = {
        'pid_params': pid_params['aileron'], 
        'writer': None,
    }

    agent_spec_aileron_PID = AgentSpec('aileron', 'PID', ['aileron'], params_aileron_pid_agent)

    #for the learning agents, a standard parameter set will do; the details will be learned
    params_DDPG_MADDPG_agent = {
        'writer': None,
    }

    agent_spec_elevator_DDPG = AgentSpec('elevator', 'DDPG', ['elevator'], params_DDPG_MADDPG_agent)

    agent_spec = [agent_spec_aileron_PID, agent_spec_elevator_DDPG]

    agent_classes_dict = {
        'PID': PID_AgentTrainer,
        'MADDPG': DDPG_AgentTrainer,
        'DDPG': MADDPG_AgentTrainer
    }

    env = MiniEnv(task_list)
    ag_container = AgentContainer.init_from_env(task_list, agent_spec, agent_classes_dict)
    ag_container.save_agent_container_data('./test_save/')

    restored_cont = AgentContainer.init_from_save('./test_save/agent_container.pickle')
    restored_cont.save_agent_container_data('./test_save/')


    print('ferti')



#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import gym
from gym.spaces import Box
import numpy as np

import types
from typing import Type, Tuple, Dict, List, Sequence, NamedTuple, Optional, Union

from collections import namedtuple

from gym_jsbsim.utils import aggregate_gym_boxes
from gym_jsbsim.agents.AgentTrainer import Experience, AgentTrainer, PID_AgentTrainer, DDPG_AgentTrainer, MADDPG_AgentTrainer

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

        self.init_dict = {
            'task_list_n': task_list_n,
            'agents_m': agents_m,
            'mapping_dict': mapping_dict,
        }
        
        self.task_list_n = task_list_n
        self.agents_m = agents_m
        self.mapping_dict = mapping_dict

        
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
        self.task_action_space_n = list(map(lambda t: t.get_action_space(), self.task_list_n))
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
    
    def remember(self, obs_n, actions_n, rewards_n, next_obs_n, dones_n):
        """
        Forwards the experience to the agents for pushing it into their replay buffers.

        TODO: A centralized replay buffer comes into mind. Check implications
        """
        exper_trans_m = list(map(self._get_per_agent_data, (obs_n, actions_n, rewards_n, next_obs_n, dones_n))) #returns an 5*m object
        # aggregate the rewards
        exper_trans_m[2] = [self.agents_m[i].rwd_aggregator(rwd_list) for i, rwd_list in enumerate(exper_trans_m[2])]
        #aggregate the dones
        exper_trans_m[4] = [any(done_list) for done_list in exper_trans_m[4]]
        experience_m = [Experience(*e) for e in list(zip(*exper_trans_m))]    # first transpose to m*5 object and then put in Experience

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
        task_actions = [np_inp[self.task_to_agent_action_idxs[t_idx]] for t_idx in range(len(self.task_list_n))]
        return task_actions

    @classmethod
    def init_from_env(cls, env: 'JsbSimEnv_multi_agent', agent_spec: List[AgentSpec], 
                      agent_classes_dict: Dict['str', 'Class'], arglist) -> 'AgentContainer':
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

        task_list_n = env.task_list
        task_names_n = [t.name for t in task_list_n]
        action_spaces_n = [t.get_action_space() for t in task_list_n]
        obs_spaces_n = [t.get_state_space() for t in task_list_n]
        
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
        agent_idx_per_task = np.full( shape=len(task_list_n), fill_value=-1, dtype=np.int32) 
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
                'agent_interaction_frequency': arglist.interaction_frequency
            }
            agent_init_dict.update(aspec.parameters)
            agent_init_dict_m.append(agent_init_dict)

            new_agent = agent_classes_dict[aspec.agent_type](**agent_init_dict)
            agents_m.append(new_agent)

        container_dict = {
            'task_list_n': task_list_n, 
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
            self.action_space = action_space
            self.state_space = state_space
        def get_action_space(self):
            return self.action_space
        def get_state_space(self):
            return self.state_space
    class MiniAgent():
        def __init__(self, name, action_width):
            self.name = name
            self.action_width = action_width
        def get_action(self, obs, add_exploration_noise = False):
            return np.full( shape=self.action_width, fill_value=self.action_width, dtype=np.float32)
        def store_experience(self, experience):
            print(f'{self.name} stored {experience}')
        def rwd_aggregator(self, rwd_list):
            return rwd_list.sum()
    class MiniEnv():
        def __init__(self, task_list):
            self.task_list = task_list
    class AG():
        def __init__(self, **kwargs):
            self.name = kwargs['name']
            print(f'{self.__class__.__name__} instantiated with:')
            for key, value in kwargs.items(): 
                print ("%s = %s" %(key, value)) 
    class PID_Agent(AG):
        ...
    class DDPG_Agent(AG):
        ...
    class MADDPG_Agent(AG):
        ...

    pid = PID_Agent(hallo='hallo', test='test', welt='welt', name='testname')

    task_list = [
        MiniTask('t1', Box(low=np.array([-1,-2,-3]), high=np.array([1,1,1])), Box(low=np.array([-1]), high=np.array([1]))),
        MiniTask('t2', Box(low=np.array([-1]), high=np.array([1])), Box(low=np.array([-2,-2]), high=np.array([2,2]))),
        MiniTask('t3', Box(low=np.array([-1,-2]), high=np.array([1,12])), Box(low=np.array([-3, -3, -3]), high=np.array([3,3,3]))),
        MiniTask('t4', Box(low=np.array([-1,-10,-12]), high=np.array([1,0,9])), Box(low=np.array([-4,-4,-4,-4]), high=np.array([4,4,4,4]))),
        MiniTask('t5', Box(low=np.array([-1]), high=np.array([1])), Box(low=np.array([-5,-5,-5,-5,-5]), high=np.array([5,5,5,5,5])))
    ]

    agent_spec = [
        AgentSpec('ag1','DDPG', ['t1', 't3'], {'hallo':'hallo', 'DDPG':'DDPG', 'ag_nr':1}),
        AgentSpec('ag3','MADDPG', ['t2', 't4'], {'hallo':'hallo', 'MADDPG':'MADDPG', 'ag_nr':3}),
        AgentSpec('ag2','PID', ['t5'], {'hallo':'hallo', 'PID':'PID', 'ag_nr':2}),
    ]

    agent_classes_dict = {
        'PID': PID_Agent,
        'MADDPG': MADDPG_Agent,
        'DDPG': DDPG_Agent
    }

    agent_classes_dict['MADDPG'](hello='world', name='hello')

    env = MiniEnv(task_list)
    ag_container = AgentContainer.init_from_env(env, agent_spec, agent_classes_dict)

    agents_m = [
        MiniAgent('ag1', 3),
        MiniAgent('ag2', 6),
        MiniAgent('ag3', 1)
    ] 

    mapping_dict = {
        'ag1': ['t1'],
        'ag3': ['t2'],
        'ag2': ['t5', 't3'],
    }

    a_cont = AgentContainer(task_list, agents_m, mapping_dict)

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

    task_actions_n = a_cont.get_actions(obs_n)

    a_cont.remember(obs_n, task_actions_n, rwd_n, obs_n, done_n)


    print('ferti')



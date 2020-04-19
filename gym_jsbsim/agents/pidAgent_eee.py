from .simple_pid import PID

import gym
import numpy as np
from abc import ABC, abstractmethod

from collections import namedtuple
from typing import List, Tuple

class AgentTrainer(ABC):
    """ 
    Similar to the AgentTrainer interface defined in https://github.com/openai/maddpg/blob/master/maddpg/__init__.py

    """
    @abstractmethod
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplementedError()

    @abstractmethod
    def action(self, obs: np.ndarray, add_exploration_noise=False) -> np.ndarray:
        """ determines the action to take from the observation.

        :param obs: The observation to use for action determination
        :param add_exploration_noise = False: flag to determine whether exploration noise shall be added to the calculated action
        """
        raise NotImplementedError()

    @abstractmethod
    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        """ Stores the transition tupel (s,a,r,s',done) to the agent's private replay buffer.

        It is up to the agent to classify the quality of the transition to e. g. use a priority replay buffer.
        However, it is important, that all agents in a multi-agent setup use the training_step%buf_size as an index to
        the stored transitions, so that the entire state of each step can be extracted form the observations of all agents.
        """
        raise NotImplementedError()

    @abstractmethod
    def preupdate(self):
        """ Any (clean-up) actions to be performed before the training step.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, agents: List['AgentTrainer'], t: int, own_idx: int):
        """ The training step of the agent.

        The other agents can be queried for their experience to reconstruct the entire state in multi agent setups.

        :param agents: The list of all agents taking part in the multi-agent setup. 
        :param t: The current training step. May be used for delayed update of the target networks
        :param own_idx: The own index of the agent in the agents list
        """
        raise NotImplementedError()

    def reset_notifier(self):   #TODO: get rid of this workaround if possible. Only needed for PID agents
        """Called on environment reset. 
        
        Is needed for PID-Agents. Should not be needed for (MA)DDPG-Agents.

        Overwrite at your convevnience.
        """
        pass


# a parameter set for a PID controller
PidParameters = namedtuple('PidParameters', ['Kp', 'Ki', 'Kd'])

class PID_Agent(AgentTrainer):
    """ An agent that realizes a PID controller.

    The PID control agent can be used as a Benchmark.
    The PID control Agent connects to a ControlGUI.py application to change the parameters interactively.
    """

    def __init__(self, name: str, pid_params: PidParameters, act_space: gym.Space, agent_interaction_freq=5):
        self.name = name
        
        self.inverted = True if pid_params.Kp <0 else False
        self.controller = PID(sample_time=None, 
                    Kp=pid_params.Kp, 
                    Ki=pid_params.Ki,
                    Kd=pid_params.Kd, 
                    output_limits=(act_space.low[0], act_space.high[0]))
        self.dt = 1.0/agent_interaction_freq    #the step time between two agent interactions in [sec] (for the PID controller)

    def reset_notifier(self):
        """when changing setpoint or when the task s reset, the PID-integrator shall be reset
        """
        self.controller.reset()

    def action(self, obs: np.ndarray, add_exploration_noise=False) -> np.ndarray:
        error = obs[0]  #for the PID controller, the control deviation shall be the first element of the [obs]; all others wil be ignored
        #using the errors keeps the setpoint constantly at 0; The errors should vanish
        control_out = self.controller(error, dt=self.dt) 
        return [control_out]

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        #PID agent doesn't learn anything and hence doesn't need any experience
        pass

    def preupdate(self):
        #PID agent doesn't learn anything and hence doesn't need any experience
        pass

    def update(self, agents, t, own_idx):
        #PID agent doesn't learn anything and hence doesn't need any experience
        return None

from gym_jsbsim.learn.ddpg_torch_eee import Agent_Single
class SingleDDPG_Agent(AgentTrainer):
    """ 
    Similar to the AgentTrainer interface defined in https://github.com/openai/maddpg/blob/master/maddpg/__init__.py

    """

    def __init__(self, name, lr_actor, lr_critic, own_input_shape, action_space, tau, gamma=0.99,
                 max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64, chkpt_dir='tmp/ddpg', chkpt_postfix='', noise_sigma = 0.15, noise_theta = 0.2):
        self.name = name
        self.agent = Agent_Single(lr_actor, lr_critic, own_input_shape, action_space, tau, gamma=gamma,
                 max_size=max_size, layer1_size=layer1_size,
                 layer2_size=layer2_size, batch_size=batch_size, chkpt_dir=chkpt_dir+'_'+self.name, chkpt_postfix=chkpt_postfix, noise_sigma = noise_sigma, noise_theta = noise_theta)
                 
    def action(self, obs: np.ndarray, add_exploration_noise=False) -> np.ndarray:
        """ determines the action to take from the observation.

        :param obs: The observation to use for action determination
        :param add_exploration_noise = False: flag to determine whether exploration noise shall be added to the calculated action
        """
        return self.agent.choose_action(obs, add_exploration_noise=add_exploration_noise)

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        """ Stores the transition tupel (s,a,r,s',done) to the agent's private replay buffer.

        It is up to the agent to classify the quality of the transition to e. g. use a priority replay buffer.
        However, it is important, that all agents in a multi-agent setup use the training_step%buf_size as an index to
        the stored transitions, so that the entire state of each step can be extracted form the observations of all agents.
        """
        return self.agent.remember(obs = obs, action = act, reward = rew, new_obs = new_obs, done = done)

    def preupdate(self):
        """ Any (clean-up) actions to be performed before the training step.
        """
        pass

    def update(self, agents: List['AgentTrainer'], t: int, own_idx):
        """ The training step of the agent.

        The other agents can be queried for their experience to reconstruct the entire state in multi agent setups.

        :param agents: ignored for SingleDDPG_Agent
        :param t: ignored for SingleDDPG_Agent
        """
        return self.agent.learn(agents, own_idx)

    def reset_notifier(self):   #TODO: get rid of this workaround if possible. Only needed for PID agents
        """Called on environment reset. 
        
        Resets the noise source like in the original DDPG paper
        """
        return self.agent.noise.reset()

from gym_jsbsim.learn.ddpg_torch_eee import Agent_Multi
class MultiDDPG_Agent(SingleDDPG_Agent):
    """ 
    Similar to the AgentTrainer interface defined in https://github.com/openai/maddpg/blob/master/maddpg/__init__.py

    """

    def __init__(self, name, lr_actor, lr_critic, own_input_shape, action_space, 
                 other_input_shapes: List[Tuple], other_actions_shapes: List[Tuple],
                 tau, gamma=0.99,
                 max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64, chkpt_dir='tmp/ddpg', chkpt_postfix='', noise_sigma = 0.15, noise_theta = 0.2):
        self.name = name

        self.agent = Agent_Multi(lr_actor, lr_critic, own_input_shape, action_space=action_space, tau = tau, 
                 other_input_shapes = other_input_shapes,
                 other_actions_shapes = other_actions_shapes,
                 gamma=gamma,
                 max_size=max_size, layer1_size=layer1_size,
                 layer2_size=layer2_size, batch_size=batch_size, 
                 chkpt_dir=chkpt_dir+'_'+self.name, chkpt_postfix=chkpt_postfix, 
                 noise_sigma = noise_sigma, noise_theta = noise_theta)
                 
    def update(self, agents: List['AgentTrainer'], t: int, own_idx):
        """ The training step of the agent.

        The other agents can be queried for their experience to reconstruct the entire state in multi agent setups.

        :param agents: ignored for SingleDDPG_Agent
        :param t: ignored for SingleDDPG_Agent
        """
        return self.agent.learn(agents, own_idx)


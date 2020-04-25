import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!


import numpy as np
import torch as T
import torch.nn.functional as F

from abc import ABC, abstractmethod

from gym.spaces import Box
from collections import namedtuple
from typing import List, Tuple, Dict

from gym_jsbsim.helper import ReplayBuffer, OUNoise
from .networks import ActorNetwork, CriticNetwork
from gym_jsbsim.utils import soft_update

Experience = namedtuple('Experience', ['obs', 'act', 'rew', 'next_obs', 'done'])
# a parameter set for a PID controller
PidParameters = namedtuple('PidParameters', ['Kp', 'Ki', 'Kd'])


class AgentTrainer(ABC):
    """ 
    Similar to the AgentTrainer interface defined in https://github.com/openai/maddpg/blob/master/maddpg/__init__.py

    """
    def __init__(self, name:str, obs_space: Box, act_space: Box, buf_len:int = 1000000, train_steps= 1, agent_interaction_freq = 5, writer: 'SummaryWriter' = None, **kwargs):
        self.name = name
        self.buf_len = buf_len
        self.type = 'UNSPECIFIED'
        self.train_steps = train_steps
        self.writer = writer    #don't save the writer on disc, inject a new one if needed

        self.act_space = act_space
        self.act_scale =  np.array((act_space.high - act_space.low) / 2., dtype=np.float32) #to scale the action output from -1...1 into the range from low...high
        self.act_shift =  np.array((act_space.high + act_space.low) / 2., dtype=np.float32) #to scale the action output from -1...1 into the range from low...high
        self.act_scale_t =  T.tensor((act_space.high - act_space.low) / 2., dtype=T.float).to('cuda:0' if T.cuda.is_available() else 'cpu') #to scale the action output from -1...1 into the range from low...high
        self.act_shift_t =  T.tensor((act_space.high + act_space.low) / 2., dtype=T.float).to('cuda:0' if T.cuda.is_available() else 'cpu') #to scale the action output from -1...1 into the range from low...high

        self.agent_dict = {
            'name': name,
            'buf_len': buf_len,
            'obs_space': obs_space,
            'act_space': act_space,
            'type': self.type,
            'train_steps': self.train_steps,
            'agent_interaction_freq': agent_interaction_freq
        }
        self.dt = 1.0/agent_interaction_freq    #the step time between two agent interactions in [sec] (for the PID controller)

        self.replay_buffer = ReplayBuffer(buf_len, obs_space.shape, act_space.shape)

        #default values for noise
        self.scaled_noise_sigma = kwargs.get('noise_sigma', 0.15)
        self.scaled_noise_theta = kwargs.get('noise_theta', 0.2)
        self.scaled_noise = OUNoise(mu=self.act_shift,sigma=self.scaled_noise_sigma, theta=self.scaled_noise_theta, dt = self.dt, scaling=self.act_scale)

    @abstractmethod
    def get_action(self, obs: np.ndarray, add_exploration_noise=False) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_target_action_t(self, obs: 'Tensor') -> 'Tensor':
        """
        :return: Returns the Tensor of actions predicted by the target network of the agent given the observation tensor. 
        """
        raise NotImplementedError()

    def rwd_aggregator(self, rwd_list):
        """
        If more than one task is associated to an agent, the rewards must be aggregated somehow.

        Overwrite at your convenience for special agents
        """
        return rwd_list.sum()

    def store_experience(self, experience: Experience):
        self.replay_buffer.store_transition(*experience)
    
    def retrieve_experience(self, batch_idxs: List[int])-> Tuple:
        """
        :param batch_idxs: a list of (random) indices to retrieve the stored samples from
        :return: Returns a Tuple(obs_b, actions_b, rewards_b, obs_next_b, terminal_b) with lists of len(batch_idxs) entries
        """
        # obs_n, actions_n, rewards_n, obs_next_n, terminal_n
        return self.replay_buffer.get_samples_from_buffer(batch_idxs)

    def preupdate(self):
        """ Any (clean-up) actions to be performed before the training step.
        """
        pass

    @abstractmethod
    def _to_eval_mode(self):
        """ switch agents networks to eval() mode
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, agents_m: List['AgentTrainer'], own_idx: int):
        """ The training step of the agent.

        The other agents can be queried for their experience to reconstruct the entire state in multi agent setups.

        :param agents: The list of all agents taking part in the multi-agent setup. 
        :param own_idx: The own index of the agent in the agents list
        """
        raise NotImplementedError()

    @abstractmethod
    def get_agent_state_params(self) -> Dict:
        """
        :return: all parameters in a dict which are needed to restore the agent
        """
        self.agent_dict.update({
            'train_steps': self.train_steps,
        })

    @classmethod
    @abstractmethod
    def restore_saved_agent(self, agent_params: Dict) -> 'AgentTrainer':
        """
        :param agent_params: the dictionary of parameters to restore an agent
        :return: an AgentTrainer instance with restored state
        """


class PID_AgentTrainer(AgentTrainer):
    def __init__(self, name:str, obs_space: Box, act_space: Box, pid_params: PidParameters, agent_interaction_freq:float = 5, buf_len:int = 1000000, writer:'SummaryWriter' = None, **kwargs):
        super().__init__(name, obs_space, act_space, buf_len, agent_interaction_freq = agent_interaction_freq, writer = writer, **kwargs)
        self.type = 'PID'
        self.pid_params = pid_params
        # signs are in line with the formerly used PID.py module. Therefore the minus-signs
        self.pid_tensor_t = T.tensor([-self.pid_params.Kp, -self.pid_params.Ki, +self.pid_params.Kd], dtype=T.float, requires_grad=False).to('cuda:0' if T.cuda.is_available() else 'cpu')


        if act_space.shape[-1] != 1:
            raise ValueError('Incorrect act_space: action space for PID_AgentTrainer must be Box(1,).')

        if obs_space.shape[-1] < 3:
            raise ValueError('Incorrect obs_space: observation space for PID_AgentTrainer must be at least Box(3,). (more values are allowed, but will be ignored)')

        self.agent_dict.update({
            'pid_params': pid_params, 
        })
        self.inverted = True if pid_params.Kp <0 else False

    def get_action(self, observation: np.ndarray, add_exploration_noise=False) -> np.ndarray:
        
        # name the values
        error = observation[0]  #for the PID controller, the order of the obs is relevant!
        derivative = observation[1]
        integral = observation[2]
        
        #using the error instead of the current value keeps the PID-setpoint constantly at 0; The errors should vanish

        # compute output
        # signs are in line with the formerly used PID.py module. Therefore the minus-signs
        mu_np = np.array([- self.pid_params.Kp * error - self.pid_params.Ki*integral + self.pid_params.Kd * derivative])
        mu_np = np.clip(mu_np, -1, 1) #to have PID-output in the range of [-1,1]
        mu_np = mu_np * self.act_scale + self.act_shift #scale to the output range given by action_space

        #alternatively, we could call 
        # mu_np = self.get_target_action(T.tensor(obs)).numpy()
        # however the conversion to tensors seems not to be worth the cost for such a simple forward pass of a PID
                
        if add_exploration_noise:
            mu_np = mu_np + self.scaled_noise()    #noise is scaled to action_space

        mu_np = np.clip(mu_np, self.act_space.low, self.act_space.high) #to have PID-output in the range of the action_space
        return mu_np

    def get_target_action_t(self, obs_t: 'Tensor') -> 'Tensor':
        """
        in the PID controller, it just calculates the output like always, but as a tensor
        """

        mu_t = obs_t * self.pid_tensor_t
        mu_t = T.sum(mu_t, dim=1)
        T.clamp_(mu_t, -1, 1) #to have PID-output in the range of [-1,1]
        mu_t = mu_t * self.act_scale_t +  self.act_shift_t  #to have PID-output in the range of the action_space

        return mu_t.unsqueeze(1)    #to get a Tesnor of shape ([batch_size, 1]) instead of ([batch_size])

    def _to_eval_mode(self):
        pass    #nothing to do for PID agent

    def train(self, agents_m: List['AgentTrainer'], own_idx: int):
        """ The training step of the agent.

        The other agents can be queried for their experience to reconstruct the entire state in multi agent setups.

        :param agents: The list of all agents taking part in the multi-agent setup. 
        :param own_idx: The own index of the agent in the agents list
        """
        pass    #nothing to do as the PID does not learn anything

    def get_agent_state_params(self) -> Dict:
        """
        :return: all parameters in a dict which are needed to restore the agent
        """
        return self.agent_dict  #for the stateless PID trainer this is pleasingly effortless

    @classmethod
    def restore_saved_agent(cls, agent_params: Dict) -> AgentTrainer:
        """
        :param agent_params: the dictionary of parameters to restore an agent
        :return: an AgentTrainer instance with restored state
        """
        agent = cls(**agent_params)
        return agent

class MADDPG_AgentTrainer(AgentTrainer):
    def __init__(self, name:str, obs_space: Box, act_space: Box, critic_state_space: Box, buf_len:int = 1000000, agent_interaction_freq = 5,
                train_steps = 1, writer:'SummaryWriter' = None,
                lr_actor = 1e-4, lr_critic = 1e-3,
                tau=0.001, gamma=0.99,
                layer1_size=400, layer2_size=300, batch_size=64, 
                noise_sigma = 0.15, noise_theta = 0.2,
                **kwargs):
        """
        :parm name: The name of the agent
        :param obs_space: the own observation space
        :param act_space: the own action space
        :param critic_state_space: critic_state_space must be the concatenation of [own_obs_space | other_obs_spaces | other_action_spaces]
        :param buf_len: size of replay buffer
        """
        super().__init__(name, obs_space, act_space, buf_len, train_steps, agent_interaction_freq, writer, noise_sigma = noise_sigma, noise_theta = noise_theta, **kwargs)
        self.type = 'MADDPG'
        self.calculate_grad_norms = kwargs.get('calculate_grad_norms', False)   #don't save to disc
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.agent_dict.update({
            'critic_state_space': critic_state_space, 
            'lr_actor': lr_actor,
            'lr_critic': lr_critic, 
            'tau': tau, 
            'gamma': gamma,
            'layer1_size': layer1_size,
            'layer2_size': layer2_size, 
            'batch_size': batch_size, 
            'noise_sigma': noise_sigma, 
            'noise_theta': noise_theta
        })

        #instantiate the networks
        # in pytorch multi-dimensional inputs are separated along the last coordinate, https://pytorch.org/docs/stable/nn.html#linear, https://stackoverflow.com/a/58591606/2682209
        self.actor = ActorNetwork(lr_actor, obs_space.shape[-1], layer1_size, layer2_size, act_space=act_space)
        self.critic = CriticNetwork(lr_critic, critic_state_space.shape[-1], layer1_size, layer2_size, n_actions=act_space.shape[-1])
        self.target_actor = ActorNetwork(lr_actor, obs_space.shape[-1], layer1_size, layer2_size, act_space=act_space, is_target_network=True)
        self.target_critic = CriticNetwork(lr_critic, critic_state_space.shape[-1], layer1_size,layer2_size, n_actions=act_space.shape[-1], is_target_network=True)

        #load existing state parameters into the networks, if they are given
        actor_state = kwargs.get('actor_state', None)
        if actor_state:     #use exisiting parameters, if available
            self.actor.load_state_dict(actor_state)

        critic_state = kwargs.get('critic_state', None)
        if critic_state:     #use exisiting parameters, if available
            self.critic.load_state_dict(critic_state)

        target_actor_state = kwargs.get('target_actor_state', None)
        if target_actor_state:     #use exisiting parameters, if available
            self.actor.load_state_dict(target_actor_state)
        else:
            soft_update(self.actor, self.target_actor, tau=1)   #with tau=1 the target net is updated entirely to the base network

        target_critic_state = kwargs.get('target_critic_state', None)
        if target_critic_state:     #use exisiting parameters, if available
            self.target_critic.load_state_dict(target_critic_state)
        else:
            soft_update(self.critic, self.target_critic, tau=1)   #with tau=1 the target net is updated entirely to the base network

        print(f'self.actor: \n{self.actor}')
        print(f'self.critic: \n{self.critic}')

    def get_action(self, observation, add_exploration_noise = False):
        self.actor.eval()   #don't calc statistics for layer normalization in action selection
        observation_t = T.tensor(observation, dtype=T.float).to(self.actor.device)    #convert to Tensor, move to GPU if available
        mu = self.actor.forward(observation_t)
        self.actor.train()  #switch to training mode
        mu_np = mu.cpu().detach().numpy()
        if add_exploration_noise:
            mu_np = mu_np + self.scaled_noise()

        return mu_np

    def get_target_action_t(self, obs_t: T.Tensor) -> T.Tensor:
        """
        in the PID controller, it just calculates the output like always, but as a tensor
        """
        self.target_actor.eval()    #don't think, that it will ever be in train() mode, but for the sake of certainty
        obs_t = obs_t.to(self.target_actor.device)  #move to GPU if possible
        mu_t = self.target_actor.forward(obs_t)

        return mu_t

    def train(self, agents_m: List['AgentTrainer'], own_idx: int):
        if self.replay_buffer.mem_cntr < self.batch_size:
            return
        
        self.train_steps += 1

        #determine the samples for minibatch
        batch_idxs = self.replay_buffer.get_batch_idxs(self.batch_size)
        # retrieve minibatch from all agents including own
        obs_m, actual_action_m, reward_m, new_obs_m, terminal_m = \
                zip(*[ag.replay_buffer.get_samples_from_buffer(batch_idxs) for ag in agents_m])
        
        #convert to pytorch tensors
        rwd_m_t = [T.tensor(rwd, dtype=T.float).to(self.critic.device) for rwd in reward_m]
        terminal_m_t = [T.tensor(dn).to(self.critic.device) for dn in terminal_m]
        new_obs_m_t = [T.tensor(new_obs, dtype=T.float).to(self.critic.device) for new_obs in new_obs_m]
        actual_action_m_t = [T.tensor(actual_action, dtype=T.float).to(self.critic.device) for actual_action in actual_action_m]
        obs_m_t = [T.tensor(obs, dtype=T.float).to(self.critic.device) for obs in obs_m]
        #the state is the concatenation of all observations (which includes doubles, but dunno)
        state_t = T.cat(obs_m_t, dim=1).to(self.critic.device)
        new_state_t = T.cat(new_obs_m_t, dim=1).to(self.critic.device)

        [ag._to_eval_mode() for ag in agents_m] #switch networks to eval mode
        
        #calculate the target action of the new state for Bellmann equation for all agents
        target_next_action_t_m = [ag.get_target_action_t(new_obs_t) for ag, new_obs_t in zip(agents_m, new_obs_m_t)]
        #create the input to the target value function (cat([new_state_m, other_next_actions), own_action)
        own_next_action_t = target_next_action_t_m.pop(own_idx)
        target_critic_next_state_t = T.cat([new_state_t, T.cat(target_next_action_t_m, dim=1)], dim=1)

        #calculate y = rew + gamma*Q_target(state, own_action)*terminal
        target_critic_next_value_t = self.target_critic.forward(target_critic_next_state_t, own_next_action_t) #calculate the target critic value of the next_state for Bellmann equation
        target_value_t = rwd_m_t[own_idx].view(self.batch_size, 1) + self.gamma*target_critic_next_value_t * terminal_m_t[own_idx].view(self.batch_size, 1)

        #calculate Q_target(state, own_action)
        own_actual_action_t = actual_action_m_t.pop(own_idx)
        critic_state_t = T.cat([state_t, T.cat(actual_action_m_t, dim=1)], dim = 1)
        critic_value_t = self.critic.forward(critic_state_t, own_actual_action_t)       #calculate the base critic value of chosen action

        self.critic.train()         #switch critic back to training mode
        self.critic.optimizer.zero_grad()
        critic_loss_t = F.mse_loss(target_value_t, critic_value_t)
        if self.writer:
            self.writer.add_scalar("critic_loss", critic_loss, global_step=self.global_step)
        critic_loss_t.backward()

        if self.calculate_grad_norms:   #TODO: 
            grad_max_m, grad_means_m = zip(*[(p.grad.abs().max().item(), (p.grad ** 2).mean().sqrt().item())  for p in list(self.critic.parameters())])
            grad_max = max(grad_max_m)
            grad_means = np.mean(grad_means_m)
            if self.writer:
                self.writer.add_scalar("critic grad_l2",  grad_means, global_step=self.train_steps)
                self.writer.add_scalar("critic grad_max", grad_max, global_step=self.train_steps)
        self.critic.optimizer.step()

        self.critic.eval()          #switch critic back to eval mode for the "loss" calculation of the actor network
        self.actor.optimizer.zero_grad()
        mu_t = self.actor.forward(obs_m_t[own_idx])
        self.actor.train()
        actor_performance_t = -self.critic.forward(critic_state_t, mu_t) # use negative performance as optimizer always does gradiend descend, but we want to ascend
        actor_performance_mean_t = T.mean(actor_performance_t)
        if self.writer:
            self.writer.add_scalar("actor_performance", -actor_performance_mean_t, global_step=self.train_steps)
        actor_performance_mean_t.backward()
        self.actor.optimizer.step()

        self._update_target_network_parameters(self.tau)    #update target to base networks with standard tau
        
    def _to_eval_mode(self):
            self.target_actor.eval()    #switch target networks to eval mode
            self.target_critic.eval()
            self.critic.eval()          #switch critic to eval mode
                
    def _update_target_network_parameters(self, tau):
        soft_update(self.critic, self.target_critic, tau)
        soft_update(self.critic, self.target_critic, tau)
    
    def get_agent_state_params(self) -> Dict:
        """
        :return: all parameters in a dict which are needed to restore the agent
        """
        #retrieve the network parameters and add them to the agent_dict
        self.agent_dict.update({
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'target_actor_state': self.target_actor.state_dict(),
            'target_critic_state': self.target_critic.state_dict(),
        })

        #update the training_steps in agent_dict
        self.agent_dict.update({
            'train_steps': self.train_steps,
        })

        return self.agent_dict

    @classmethod
    def restore_saved_agent(cls, agent_params: Dict, writer: 'SummaryWriter' = None, pristine_networks: bool = False) -> AgentTrainer:
        """
        :param agent_params: the dictionary of parameters to restore an agent
        :return: an AgentTrainer instance with restored state
        """
        if pristine_networks:
            agent_params.pop('actor_state', None)
            agent_params.pop('critic_state', None)
            agent_params.pop('target_actor_state', None)
            agent_params.pop('target_critic_state', None)
        
        if writer:  #we are getting a summary writer, so let's use it
            agent_params.update({
                'writer': writer,
            })

        agent = cls(**agent_params)
        return agent

class DDPG_AgentTrainer(MADDPG_AgentTrainer):
    def __init__(self, name:str, obs_space: Box, act_space: Box, buf_len:int = 1000000, agent_interaction_freq = 5,
                train_steps = 1, writer:'SummaryWriter' = None,
                lr_actor = 1e-4, lr_critic = 1e-3,
                tau=0.001, gamma=0.99,
                layer1_size=400, layer2_size=300, batch_size=64, 
                noise_sigma = 0.15, noise_theta = 0.2,
                **kwargs):
        critic_state_space = kwargs.pop('critic_state_space', obs_space)    #if it was contained in the params, we take it otherwise, we know ist
        if critic_state_space != obs_space:
            raise ValueError('In DDPG_AgentTrainer, (critic_state_space == obs_space) must hold, or the parameter must not be given')
        
        super().__init__(name, obs_space, act_space, critic_state_space = obs_space, buf_len = buf_len, agent_interaction_freq = agent_interaction_freq,
                train_steps = train_steps, writer = writer,
                lr_actor = lr_actor, lr_critic = lr_critic,
                tau=tau, gamma=gamma,
                layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size, 
                noise_sigma = noise_sigma, noise_theta = noise_theta,
                **kwargs)

        self.type = 'DDPG'

    def train(self, agents_m: List['AgentTrainer'], own_idx: int):
        #agents_m and own_idx are ignored, as this is only the DDPG algo, not the MADDPG
        if self.replay_buffer.mem_cntr < self.batch_size:
            return
        
        self.train_steps += 1

        #determine the samples for minibatch
        batch_idxs = self.replay_buffer.get_batch_idxs(self.batch_size)
        # retrieve minibatch from own replay buffer
        obs, actual_action, rwd, new_obs, terminal = self.replay_buffer.get_samples_from_buffer(batch_idxs)
        
        #convert to pytorch tensors
        obs_t = T.tensor(obs, dtype=T.float).to(self.critic.device)
        actual_action_t = T.tensor(actual_action, dtype=T.float).to(self.critic.device)
        rwd_t = T.tensor(rwd, dtype=T.float).to(self.critic.device)
        new_obs_t = T.tensor(new_obs, dtype=T.float).to(self.critic.device)
        terminal_t = T.tensor(terminal).to(self.critic.device)

        self.target_actor.eval()    #switch target networks to eval mode
        self.target_critic.eval()
        self.critic.eval()          #switch critic to eval mode
        
        #calculate the target of the new state for Bellmann equation
        target_next_actions = self.target_actor.forward(new_obs_t) 
        #calculate the target critic value of the new_state for Bellmann equation
        target_critic_next_value = self.target_critic.forward(new_obs_t, target_next_actions)
        #calculate the base critic value of chosen action
        critic_value = self.critic.forward(obs_t, actual_action_t)

        target_value = rwd_t.view(self.batch_size, 1) + self.gamma*target_critic_next_value*terminal_t.view(self.batch_size, 1)


        self.critic.train()         #switch critic back to training mode
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target_value, critic_value)
        if self.writer:
            self.writer.add_scalar("critic_loss", critic_loss, global_step=self.train_steps)
        critic_loss.backward()

        if self.calculate_grad_norms:   #TODO: 
            grad_max_n, grad_means_n = zip(*[(p.grad.abs().max().item(), (p.grad ** 2).mean().sqrt().item())  for p in list(self.critic.parameters())])
            grad_max = max(grad_max_n)
            grad_means = np.mean(grad_means_n)
            if self.writer:
                self.writer.add_scalar("critic grad_l2",  grad_means, global_step=self.train_steps)
                self.writer.add_scalar("critic grad_max", grad_max, global_step=self.train_steps)
        
        self.critic.optimizer.step()

        #now train the actor
        self.critic.eval()          #switch critic back to eval mode for the "loss" calculation of the actor network
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(obs_t)
        self.actor.train()          #switch actor back to training mode
        actor_performance = -self.critic.forward(obs_t, mu) # use negative performance as optimizer always does gradiend descend, but we want to ascend
        actor_performance = T.mean(actor_performance)
        if self.writer:
            self.writer.add_scalar("actor_performance", -actor_performance, global_step=self.global_step)
        actor_performance.backward()
        self.actor.optimizer.step()

        self._update_target_network_parameters(self.tau)    #update target to base networks with standard tau
        

if __name__ == '__main__':

    from gym_jsbsim.utils import aggregate_gym_boxes

    box_1 = Box(np.array([-1]), np.array([1]))
    box_2 = Box(np.array([-2]*2), np.array([2]*2))
    box_3 = Box(np.array([-3]*3), np.array([3]*3))
    box_4 = Box(np.array([-4]*4), np.array([4]*4))

    pid_params = {'aileron':  PidParameters(3.5e-2,    1e-2,   0.0),
                  'elevator': PidParameters( -5e-2, -6.5e-2, -1e-3)}

    try:
        pid = PID_AgentTrainer('pid_test', box_3, box_2, pid_params['aileron'], hello='hello', world= 'world')
    except Exception as e:
        print('this should happen as the action space was incorrect', e)

    pid = PID_AgentTrainer('pid_test', box_3, box_1, pid_params['elevator'], hello='hello', world= 'world')

    act = pid.action(np.array([2]*3), True)

    obs_100 = np.array([np.array([2]*3)]*100)
    obs_100_t = T.tensor(obs_100, dtype=T.float).to('cuda:0' if T.cuda.is_available() else 'cpu')
    act_100_t = pid.get_target_action_t(obs_100_t)

    params = pid.get_agent_state_params()
    pid_restore = PID_AgentTrainer.restore_saved_agent(params)
    params = pid_restore.get_agent_state_params()
    pid_restore = PID_AgentTrainer.restore_saved_agent(params)

    maddpg_default = MADDPG_AgentTrainer('maddpg_teste', box_3, box_2, aggregate_gym_boxes([box_3, box_4, box_3, box_1, box_1]))
    params = maddpg_default.get_agent_state_params()

    maddpg_restore = MADDPG_AgentTrainer.restore_saved_agent(params)

    # 
    maddpg = MADDPG_AgentTrainer('maddpg_ second_test', box_3, box_2, aggregate_gym_boxes([box_3, box_4, box_3, box_1, box_1]),
                lr_actor = 1e-4, lr_critic = 1e-3,
                tau=0.001, gamma=0.95,
                layer1_size=200, layer2_size=200, batch_size=128, 
                noise_sigma = 0.25, noise_theta = 0.2,
                hello='hello', world= 'world')
    
    params = maddpg.get_agent_state_params()
    maddpg = MADDPG_AgentTrainer.restore_saved_agent(params, pristine_networks= True)

    ddpg = DDPG_AgentTrainer('ddpg_teste', box_4, box_1)
    params = ddpg.get_agent_state_params()

    ddpg_restore = DDPG_AgentTrainer.restore_saved_agent(params)

    obs_pid = np.array([2]*3)
    obs_maddpg = np.array([3]*3)
    obs_ddpg = np.array([4]*4)

    maddpg_2 = MADDPG_AgentTrainer('ddpg_teste', box_4, box_1, aggregate_gym_boxes([box_4, box_3, box_3, box_2, box_1]))


    trainers = [pid, maddpg, ddpg]
    trainers = [pid, maddpg, maddpg_2]
    # trainers = [pid, pid, pid]
    # trainers = [ddpg, ddpg, ddpg]
    obs_m = [obs_pid, obs_maddpg, obs_ddpg]
    # obs_m = [obs_pid, obs_pid, obs_pid]
    # obs_m = [obs_ddpg, obs_ddpg, obs_ddpg]

    from timeit import timeit    #TODO: remove again after testing

    time = 0 
    for i in range(1000):

        act_m = [ag.action(obs, True) for ag, obs in zip(trainers, obs_m)]

        experience__m = [Experience(obs, act, 1, obs, False) for obs, act in zip(obs_m, act_m)]

        [ag.store_experience(exp) for ag, exp in zip(trainers, experience__m)]

        time += timeit(lambda: 
            [ag.train(trainers, i) for i, ag in enumerate(trainers)]
            , number=1)
    print(f'{i+1} iterations with {len(trainers)} in {time} seconds')
        
    print('fertig')

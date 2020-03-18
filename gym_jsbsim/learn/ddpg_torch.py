import os
from pathlib import Path

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)    #to save the done flags in terminal states

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done  #yields 0 in all terminal states, 1 otherwise; to multiply the value by this value; 
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]       #this list picking is from numpy; 
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])     #weight initialization according to DDPD-paper
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)  #Applies _Layer_  Normalization over a mini-batch of inputs as described in the paper Layer Normalization_ .

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        #f2 = 0.002
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 3e-3
        self.q = nn.Linear(self.fc2_dims, 1)        #the Critic output is just a single scalar Q-value;
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        #self.q.weight.data.uniform_(-f3, f3)
        #self.q.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)       #activation is done after normalization to avoid cuting off the negative end
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))    #activate the action; (this gets activated twice with relu)
                                                            #TODO: what happens to negative actions? they are entirely zeroed in this setting.
        state_action_value = F.relu(T.add(state_value, action_value))   #add state and action value together
        state_action_value = self.q(state_action_value)

        return state_action_value   #scalar vaue

    def save_checkpoint(self, name_suffix=''):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file+'_'+name_suffix)

    def load_checkpoint(self, name_suffix = ''):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file+'_'+name_suffix))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f2 = 0.002
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        #f3 = 0.004
        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        #self.mu.weight.data.uniform_(-f3, f3)
        #self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self, name_suffix=''):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file+'_'+name_suffix)

    def load_checkpoint(self, name_suffix=''):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file+'_'+name_suffix))

class Agent(object):
    def __init__(self, lr_actor, lr_critic, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64, chkpt_dir='tmp/ddpg', chkpt_postfix=''):
        #default values for noise
        self.noise_sigma = 0.15
        self.noise_theta = 0.2

        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        
        self.chkpt_postfix = '_'+chkpt_postfix if chkpt_postfix != '' else ''
        self.chkpt_dir= f"checkpoints/{chkpt_dir}/inputs_{input_dims}_actions_{n_actions}/layer1_{layer1_size}_layer2_{layer2_size}/"
        Path(self.chkpt_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"set checkpoint directory to: {self.chkpt_dir}")

        self.actor = ActorNetwork(lr_actor, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='Actor'+self.chkpt_postfix, chkpt_dir= self.chkpt_dir)
        self.critic = CriticNetwork(lr_critic, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions,
                                    name='Critic'+self.chkpt_postfix, chkpt_dir= self.chkpt_dir)

        self.target_actor = ActorNetwork(lr_actor, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         name='TargetActor'+self.chkpt_postfix, chkpt_dir= self.chkpt_dir)
        self.target_critic = CriticNetwork(lr_critic, input_dims, layer1_size,
                                           layer2_size, n_actions=n_actions,
                                           name='TargetCritic'+self.chkpt_postfix, chkpt_dir= self.chkpt_dir)

        # self.noise = OUActionNoise(mu=np.zeros(n_actions),sigma=0.15, theta=.2, dt=1/5.)
        self.noise = OUActionNoise(mu=np.zeros(n_actions),sigma=self.noise_sigma, theta=self.noise_theta, dt=1/5.)

        writer_name = f"GLIDE-DDPG_input_dims-{input_dims}_n_actions-{n_actions}_lr_actor-{lr_actor}_lr_critic-{lr_critic}_batch_size-{batch_size}"
        self.writer = SummaryWriter(comment=writer_name)

        print(self.actor)
        print(self.critic)

        self.update_network_parameters(tau=1)   #with tau=1 the target net is updated entirely to the base network
        self.global_step = 0
        self.episode_counter = 0

    def choose_action(self, observation, add_exploration_noise = True):
        self.actor.eval()   #don't calc statistics for layer normalization in action selection
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)    #convert to Tensor
        mu = self.actor.forward(observation).to(self.actor.device)
        noise = self.noise() if add_exploration_noise else 0
        # if self.writer:
        #     self.writer.add_scalar("exploration noise", noise, global_step=self.global_step)

        mu = mu + T.tensor(noise,                          #add exploration noise
                           dtype=T.float).to(self.actor.device)
        self.actor.train()  #switch to training mode
        return mu.cpu().detach().numpy()  #return actions as numpy array


    def remember(self, state, action, reward, new_state, done):
        if self.writer:
            self.writer.add_scalar("reward", reward, global_step=self.global_step)
        self.memory.store_transition(state, action, reward, new_state, done)
        if done:
            self.episode_counter += 1
        self.global_step += 1
        

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, actual_action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        actual_action = T.tensor(actual_action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()    #switch target networks to eval mode
        self.target_critic.eval()
        self.critic.eval()          #switch critic to eval mode
        target_next_actions = self.target_actor.forward(new_state)   #calculate the target action of the new statefor Bellmann equation
        target_critic_next_value = self.target_critic.forward(new_state, target_next_actions) #calculate the target critic value of the new_state for Bellmann equation
        critic_value = self.critic.forward(state, actual_action)       #calculate the base critic value of chosen action

        target_value = []
        for j in range(self.batch_size):    #TODO: this could also be done in vectorized implementation (efficiency)
            target_value.append(reward[j] + self.gamma*target_critic_next_value[j]*done[j])    #zero at the end of an episode
        target_value = T.tensor(target_value).to(self.critic.device)
        target_value = target_value.view(self.batch_size, 1)    #reshape (why?)

        self.critic.train()         #switch critic back to training mode
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target_value, critic_value)
        if self.writer:
            self.writer.add_scalar("critic_loss", critic_loss, global_step=self.global_step)
        critic_loss.backward()
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in self.critic.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1
        self.writer.add_scalar("critic grad_l2",  grad_means / grad_count, global_step=self.global_step)
        self.writer.add_scalar("critic grad_max", grad_max, global_step=self.global_step)
        self.critic.optimizer.step()

        self.critic.eval()          #switch critic back to eval mode for the "loss" calculation of the actor network
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_performance = -self.critic.forward(state, mu) # use negative performance as optimizer always does gradiend descend, but we want to ascend
        actor_performance = T.mean(actor_performance)
        if self.writer:
            self.writer.add_scalar("actor_performance", -actor_performance, global_step=self.global_step)
        actor_performance.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()    #update target to base networks with standard tau

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        """
        #Verify that the copy assignment worked correctly
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(target_critic_params)
        actor_state_dict = dict(target_actor_params)
        print('\nActor Networks', tau)
        for name, param in self.actor.named_parameters():
            print(name, T.equal(param, actor_state_dict[name]))
        print('\nCritic Networks', tau)
        for name, param in self.critic.named_parameters():
            print(name, T.equal(param, critic_state_dict[name]))
        input()
        """
    def save_models(self, name_discriminator = ''):
        self.actor.save_checkpoint(name_discriminator)
        self.target_actor.save_checkpoint(name_discriminator)
        self.critic.save_checkpoint(name_discriminator)
        self.target_critic.save_checkpoint(name_discriminator)

    def load_models(self, name_discriminator = ''):
        self.actor.load_checkpoint(name_discriminator)
        self.target_actor.load_checkpoint(name_discriminator)
        self.critic.load_checkpoint(name_discriminator)
        self.target_critic.load_checkpoint(name_discriminator)

    def reset_noise_source(self):
        self.noise.reset()

    def reduce_noise_sigma(self, sigma_factor = 1, theta_factor = 1):
        self.noise_sigma *= sigma_factor
        self.noise_theta *= theta_factor
        print('Noise set to sigma=%f, theta=%f' % (self.noise_sigma, self.noise_theta))
        self.noise = OUActionNoise(mu=np.zeros(self.n_actions),sigma=self.noise_sigma, theta=self.noise_theta, dt=1/5.)
        self.noise.reset()

    # def check_actor_params(self):
    #     current_actor_params = self.actor.named_parameters()
    #     current_actor_dict = dict(current_actor_params)
    #     original_actor_dict = dict(self.original_actor.named_parameters())
    #     original_critic_dict = dict(self.original_critic.named_parameters())
    #     current_critic_params = self.critic.named_parameters()
    #     current_critic_dict = dict(current_critic_params)
    #     print('Checking Actor parameters')

    #     for param in current_actor_dict:
    #         print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
    #     print('Checking critic parameters')
    #     for param in current_critic_dict:
    #         print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
    #     input()

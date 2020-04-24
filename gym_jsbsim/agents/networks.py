from gym.spaces import Box

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from contextlib import ExitStack    #to have a conditional with torch.no_grad() context, see https://stackoverflow.com/a/34798330/2682209

class CriticNetwork(nn.Module):
    def __init__(self, beta, n_inputs, fc1_dims, fc2_dims, n_actions, is_target_network:bool = False):
        """
        :param is_target_network: the target networks neither need an optimizer nor do they need gradient caclulation.
        """
        super(CriticNetwork, self).__init__()

        self.is_target_network = is_target_network

        self.n_inputs = n_inputs
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.n_inputs, self.fc1_dims)
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
        self.q = nn.Linear(self.fc2_dims, 1)        #the Critic output is just a single scalar Q-value (wrapped in a tensor);
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        #self.q.weight.data.uniform_(-f3, f3)
        #self.q.bias.data.uniform_(-f3, f3)

        if not is_target_network:
            self.optimizer = optim.Adam(self.parameters(), lr=beta) #no need for an optimizer in target networks
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        with T.no_grad() if self.is_target_network else ExitStack():    #see https://stackoverflow.com/a/34798330/2682209
            state_value = self.fc1(state)
            state_value = self.bn1(state_value)
            state_value = F.relu(state_value)       #activation is done after normalization to avoid cuting off the negative end
            state_value = self.fc2(state_value)
            state_value = self.bn2(state_value)

            action_value = F.relu(self.action_value(action))    #activate the action; (this gets activated twice with relu)
                                                                #TODO: what happens to negative actions? they are entirely zeroed in this setting. No, the self.action_network has weights and bias, so it can hold for it
            state_action_value = F.relu(T.add(state_value, action_value))   #add state and action value together
            state_action_value = self.q(state_action_value)

        return state_action_value   #scalar value

class ActorNetwork(nn.Module):
    def __init__(self, alpha, n_inputs: int, fc1_dims, fc2_dims, act_space:Box, is_target_network:bool = False):
        super(ActorNetwork, self).__init__()

        self.is_target_network = is_target_network  #target networks can do the forward pass with torch.no_grad()

        self.n_inputs = n_inputs
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = act_space.shape[-1]
        self.act_scale_t =  T.tensor((act_space.high - act_space.low) / 2., dtype=T.float) #to scale the action output from -1...1 into the range from low...high
        self.act_shift_t =  T.tensor((act_space.high + act_space.low) / 2., dtype=T.float) #to scale the action output from -1...1 into the range from low...high

        self.fc1 = nn.Linear(self.n_inputs, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
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

        self.act_scale_t = self.act_scale_t.to(self.device)
        self.act_shift_t = self.act_shift_t.to(self.device)
        self.to(self.device)

    def forward(self, state: 'tensor') -> 'tensor':
        with T.no_grad() if self.is_target_network else ExitStack():    #see https://stackoverflow.com/a/34798330/2682209
            x = self.fc1(state)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.mu(x)
            x = T.tanh(x)
            x = x * self.act_scale_t +  self.act_shift_t  #to have output scaled to the range of the action_space

        return x

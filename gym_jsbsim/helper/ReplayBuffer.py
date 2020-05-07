import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, actions_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.obs_memory = np.zeros((self.mem_size, *input_shape))
        self.obs_next_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, *actions_shape))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)    #to save the done flags in terminal states

    def store_transition(self, obs, action, reward, obs_next, done):
        index = self.mem_cntr % self.mem_size
        self.obs_memory[index] = obs
        self.obs_next_memory[index] = obs_next
        self.action_memory[index] = [act for act in action if act != None]  #filter out None values in the actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done  #yields 0 in all terminal states, 1 otherwise; to multiply the value by this value; 
        self.mem_cntr += 1

    def get_batch_idxs(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch_idxs = np.random.choice(max_mem, batch_size)

        return batch_idxs

    def get_samples_from_buffer(self, batch_idxs):

        obs = self.obs_memory[batch_idxs]       #this list picking is from numpy; 
        actions = self.action_memory[batch_idxs]
        rewards = self.reward_memory[batch_idxs]
        obs_next = self.obs_next_memory[batch_idxs]
        terminal = self.terminal_memory[batch_idxs]

        return obs, actions, rewards, obs_next, terminal


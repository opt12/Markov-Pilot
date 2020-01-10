import gym
import numpy as np
from collections import namedtuple


class StateSelectWrapper(gym.Wrapper):
    """
    A wrapper to select a subset of the observation space by passing a list 
    names of the original named tuple.
    The observation space of the wrapped env is adapted to the new shape.

    Pass in a list of state names which are present in the underlying named tuple of the gym_jsbsim.FlightTask. 
    The wrapper will extract the positions of the named states in the tuple/np.array. 
    np.take() is used to project the original state to the new (reduced) state.
    """
    
    def __init__(self, env, selected_state_vars:list):
        super(StateSelectWrapper, self).__init__(env)
        self.original_env = env
        #get the postitions in the original tuple
        state_names = env.get_state_property_names()
        self.selection_list = []
        #temp store for the limit values of the new observation space
        new_low, new_high = np.array([]), np.array([])

        for selection in selected_state_vars:
            idx = state_names.index(selection)  #if element is not in list, a ValueError exception is raised
            self.selection_list.append(idx)    
            new_low = np.append(new_low, self.original_env.observation_space.low[idx])
            new_high = np.append(new_high, self.original_env.observation_space.high[idx])

        #adapt the observation space
        self.observation_space = gym.spaces.Box(new_low, new_high, dtype=self.original_env.action_space.dtype)
        print("Observation space reduced to {} elements: {}".format(len(self.selection_list), selected_state_vars))


    def step(self, action):
        #perform step
        original_state, reward, done, info  = self.env.step(action)
        #select the state
        state = np.take(original_state, self.selection_list)
        return state, reward, done, info
    
    def reset(self):
        return np.take(self.original_env.reset(), self.selection_list)


    




import gym
import numpy as np
from collections import namedtuple

from gym_jsbsim.agents import PIDAgentSingleChannel, PidParameters

# a parameter set for a PID controller
PidWrapperParams = namedtuple('PidWrapperParams', ['action_name', 'error_state_name', 'pid_params'])


class PidWrapper(gym.Wrapper):
    """
    A wrapper to replace certain actions from the env.action_space with
    ordinary PID-control.
    The action space of the wrapped env is reduced by the replaced actions.

    Pass in a list of PidWrapperParams. this consists of 
    - action names (according to the env.task.action_variables.get_legal_names())
    - error_state name (according to the env.task.state_variables.get_legal_names()) 
      to specify the state variable containing the error for this PID channel
    - a PidParameters-tuple to specify the controller parameters
    The controller limits are taken from the corresponding acition_space.
    
    """
    
    def __init__(self, env, wrapped_actions: PidWrapperParams):
        super(PidWrapper, self).__init__(env)
        self.state = np.zeros(env.observation_space.shape[0])
        self.original_env = env

        try:
            self.agent_interaction_freq = env.JSBSIM_DT_HZ / env.sim_steps_per_agent_step
        except:
            print("Using default agent interaction frequency of 5Hz")
            self.agent_interaction_freq = 5
        

        self.state_names = list(map(lambda el:el.get_legal_name(), env.task.state_variables))
        self.action_names = list(map(lambda el:el.get_legal_name(), env.task.action_variables))

        self.original_actions_len = self.original_env.action_space.shape[0]
        #adjust the action_space to miss out the replaced values
        new_low, new_high = np.array([]), np.array([])

        self.error_state_idxs = []
        self.pid_controllers = []
        self.pid_positions, self.original_positions = [], []
        #unpack the input tuple list:   https://stackoverflow.com/a/7558990/2682209
        wrapped_action_names, error_state_names, pid_params = zip(*wrapped_actions)
        #get the replacment PID controllers
        for action_idx in range(self.original_actions_len):
            if (self.action_names[action_idx] in wrapped_action_names):
                #action shall be replaced
                wrap_list_idx = wrapped_action_names.index(self.action_names[action_idx])
                replaced_action_idx = action_idx
                error_state_idx     = self.state_names.index(error_state_names[wrap_list_idx])
                self.pid_positions.append(replaced_action_idx)
                self.error_state_idxs.append(error_state_idx)
                self.pid_controllers.append(PIDAgentSingleChannel(pid_params[wrap_list_idx],
                            self.original_env.action_space.low[replaced_action_idx],
                            self.original_env.action_space.high[replaced_action_idx],
                            self.agent_interaction_freq)) #TODO: PIDAgent must be single Agent..., not both
            else:
                #this is a value not to be replaced
                new_low = np.append(new_low, self.original_env.action_space.low[action_idx])
                new_high = np.append(new_high, self.original_env.action_space.high[action_idx])
                self.original_positions.append(action_idx)
        self.action_space = gym.spaces.Box(new_low, new_high, dtype=self.original_env.action_space.dtype)


    def step(self, action):
        appliedActions = np.empty(self.original_actions_len)
        # get the pid actions:
        for idx, controller in enumerate(self.pid_controllers):
            pid_action = controller.act(self.state[self.error_state_idxs[idx]])
            appliedActions[self.pid_positions[idx]] = pid_action
        for orig_position, act in zip(self.original_positions, action):
            appliedActions[orig_position] = act
        # issue the merged action to the original environment
        obs = self.original_env.step(appliedActions)
        self.state, _, _, _ = obs
        return obs
    
    def reset(self):
        self.state = self.original_env.reset()
        return self.state

    #TODO: When change_setpoints is called for the PID controlled variables, the integrator and derivative shall be reset.
    




from .simple_pid import PID
import gym
import numpy as np

from collections import namedtuple

class AgentTrainer(object):
    """ 
    Similar to the AgentTrainer interface defined in https://github.com/openai/maddpg/blob/master/maddpg/__init__.py

    """
    # def __init__(self, name, model, obs_shape, act_space, args):
    #     raise NotImplementedError()

    def action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplementedError()

    def preupdate(self):
        raise NotImplementedError()

    def update(self, agents):
        raise NotImplementedError()


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

    def action(self, obs: np.ndarray) -> np.ndarray:
        error = obs[0]  #for the PID controller, the control deviation shall be the only input
        #using the errors keeps the setpoint constantly at 0; The errors should vanish
        control_out = self.controller(error, dt=self.dt) 
        return [control_out]

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        #PID agent doesn't learn anything and hence doesn't need any experience
        pass

    def preupdate(self):
        #PID agent doesn't learn anything and hence doesn't need any experience
        pass

    def update(self, agents):
        #PID agent doesn't learn anything and hence doesn't need any experience
        pass

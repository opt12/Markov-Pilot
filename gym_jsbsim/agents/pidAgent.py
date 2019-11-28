from .agents import Agent
from .simple_pid import PID
import gym
import gym_jsbsim.properties as prp
import numpy as np

class PIDAgent(Agent):
    """ An agent that realizes a PID controller.

    The PID control agent can be used as a Benchmark.
    The PID control Agent connects to a ControlGUI.py application to change the parameters interactively.
    """
    def __init__(self, action_space: gym.Space, agent_interaction_freq=5):
        self.action_space = action_space
        self.controllers = {}
        self.controllers['pitchControl'] = PID(sample_time=None, Kp=-5e-2, Ki=-6.5e-2, Kd=-1e-3, output_limits=(-1, 1))
        self.controllers['rollControl']  = PID(sample_time=None, Kp= 3.5e-2, Ki= 1e-2,   Kd=0    , output_limits=(-1, 1))
        self.dt = 1.0/agent_interaction_freq    #the step time between two agent interactions in [sec] (for the PID controller)
    
    def getControllerNames(self):
        """
        returns the keys of the self.controllers dictionary to manipulate them via GUI
        """
        return self.controllers.keys()
    
    def setControllerParams(self, key, Kp=0, Ki=0, Kd=0, auto_mode=True):
        newTuning = np.array([Kp, Ki, Kd])
        self.controllers[key].tunings = newTuning.tolist()
        print("set tunings to: {}".format(newTuning))
        if auto_mode != self.controllers[key].auto_mode:
            self.controllers[key].auto_mode = auto_mode
        
    def act(self, state):
        #TODO: the setpoint shall be propagated from the GUI to the flight task, so this should also registe @GUI
        #using the errors keeps the setpoint constantlyat 0; The errors should vanish
        rollAngle_error_deg = state[14]     #TODO: indices should be replaced by names somehow
        glideAngle_error_deg = state[13]

        elevator = self.controllers['pitchControl'](glideAngle_error_deg, dt=self.dt)
        aileron = self.controllers['rollControl'](rollAngle_error_deg, dt=self.dt)

        # print('elevator: {} => {}'.format(glideAngle_error_deg, elevator))

        return np.array([elevator,aileron]) #we don't control the rudder for the moment
    
    def getActionNames(self):
        return('elevator', 'aileron')

    def observe(self, state, action, reward, done):
        # this agent type does not learn in response to observations
        pass


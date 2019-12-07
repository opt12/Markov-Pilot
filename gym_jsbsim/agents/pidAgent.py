from .agents import Agent
from .simple_pid import PID
import gym
import gym_jsbsim.properties as prp
import numpy as np
import threading
import rpyc
import time

from collections import namedtuple

# a parameter set for a PID controller
PidParameters = namedtuple('PidParameters', ['Kp', 'Ki', 'Kd'])

class PIDAgent(Agent):
    """ An agent that realizes a PID controller.

    The PID control agent can be used as a Benchmark.
    The PID control Agent connects to a ControlGUI.py application to change the parameters interactively.
    """
    def __init__(self, action_space: gym.Space, agent_interaction_freq=5, env = None):
        self.action_space = action_space
        self.controllers = {}
        self.pitchControlParams = {'Kp': -5e-2, 'Ki': -6.5e-2, 'Kd': -1e-3}
        self.rollControlParams = {'Kp': 3.5e-2, 'Ki': 1e-2, 'Kd': 0.0}
        self.controllers['pitchControl'] = PID(sample_time=None, 
                    Kp=self.pitchControlParams['Kp'], 
                    Ki=self.pitchControlParams['Ki'],
                    Kd=self.pitchControlParams['Kd'], output_limits=(-1, 1))
        self.controllers['rollControl']  = PID(sample_time=None, 
                    Kp=self.rollControlParams['Kp'],
                    Ki=self.rollControlParams['Ki'],
                    Kd=self.rollControlParams['Kd'], output_limits=(-1, 1))
        self.dt = 1.0/agent_interaction_freq    #the step time between two agent interactions in [sec] (for the PID controller)

        self.env = env  #this is used in conjunction with the ControlGUI.py to change setpoints and PID parameters in real time

        if env: 
            try:
                # all of this is a dirty hack, but it works.
                # When starting ControlGUI.py in anaother console, one can adjust the PID settings and the 
                # roll and glide ange setpoints
                client = RPYCClient()
                client.start()
                print("started client")
                env.reset() #only now, the sim.jsbsim property gets populated
                try:
                    client.addQuadSlider(name="pitchControl", 
                            cb = getCallback(self.env, self, 'pitchControl', 'target/glideAngle-deg', inverted = True))
                    client.setValue('pitchControl', 'sliderP',  -1*self.pitchControlParams['Kp'])   #TODO: it's only positive so far
                    client.setValue('pitchControl', 'sliderPI', -1*self.pitchControlParams['Ki'])   #TODO: it's only positive so far
                    client.setValue('pitchControl', 'sliderPD', -1*self.pitchControlParams['Kd'])   #TODO: it's only positive so far
                    client.setValue('pitchControl', 'sliderSetpoint', env.task.TARGET_GLIDE_ANGLE_DEG)   #TODO: it's only positive so far
                    client.addQuadSlider(name="rollControl", setMin = 35, setMax = -35, setRes = 1, 
                            cb = getCallback(self.env, self, 'rollControl', 'target/roll-deg'))
                    client.setValue('rollControl', 'sliderP',  self.rollControlParams['Kp'])   #TODO: it's only positive so far
                    client.setValue('rollControl', 'sliderPI', self.rollControlParams['Ki'])   #TODO: it's only positive so far
                    client.setValue('rollControl', 'sliderPD', self.rollControlParams['Kd'])   #TODO: it's only positive so far
                    client.setValue('rollControl', 'sliderSetpoint', env.task.TARGET_ROLL_ANGLE_DEG)
                except:
                    print("could not register controller widgets")
            except:
                print("GUI-Client could not cnnect to ControlGUI.py. Sorry")

    
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

class PIDAgentSingleChannel(Agent):
    """ An agent that realizes a single channel PID controller.

    The PID control agent can be used as a Benchmark.
    If a non-empty env is passed, the PID control Agent tries to connect 
    to a ControlGUI.py application to change the parameters interactively.
    """
    def __init__(self, pid_parameters: PidParameters, low_limit: float, high_limit: float, agent_interaction_freq=5.0):
        # self.pitchControlParams = {'Kp': -5e-2, 'Ki': -6.5e-2, 'Kd': -1e-3}
        # self.rollControlParams = {'Kp': 3.5e-2, 'Ki': 1e-2, 'Kd': 0.0}
        self.inverted = True if pid_parameters.Kp <0 else False
        self.controller = PID(sample_time=None, 
                    Kp=pid_parameters.Kp, 
                    Ki=pid_parameters.Ki,
                    Kd=pid_parameters.Kd, 
                    output_limits=(low_limit, high_limit))
        self.dt = 1.0/agent_interaction_freq    #the step time between two agent interactions in [sec] (for the PID controller)
    
    def act(self, error):
        #using the errors keeps the setpoint constantly at 0; The errors should vanish
        control_out = self.controller(error, dt=self.dt)

        # print('PID output: {} => {}'.format(error, control_out))

        return control_out
    
    def observe(self, state, action, reward, done):
        # this agent type does not learn in response to observations
        pass

class RPYCClient(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        conn = rpyc.connect("localhost", 12345)
        self.bgsrv = rpyc.BgServingThread(conn, self.backgroundStoppedCb)
        self.bgsrv.SLEEP_INTERVAL = 0.025   #to make the GUI more reactive
        self.bgsrvRunning = True
        self.c = conn.root
        # callback("Hallo aus der __init__")
        # self.registerCallbackBtn("from RPYCCLient", callback)

    def backgroundStoppedCb(self):
        self.bgsrvRunning = False

    def addQuadSlider(self, name, setMin = 15, setMax = -15, setRes = 0.2, cb = None, **options):
        self.c.addQuadSlider(name, setMin, setMax, setRes, cb, **options)
    
    def printMessage(self, msg):
        self.c.printMessage(msg)
    
    def setValue(self, QuadSliderKey, SliderKey, value):
        return self.c.setValue(QuadSliderKey, SliderKey, value)

    def run(self):
        while self.bgsrvRunning:
            time.sleep(1)

def getCallback(env, agent, name, propName, inverted=False, **kwargs):
    def tuningCallback(**kwargs):
        print("tuning: {}:".format(name))
        for key, value in kwargs.items(): 
            print ("%s == %s" %(key, value)) 
        #change PID params
        if inverted:
            agent.setControllerParams(name, -kwargs['valueP'], -kwargs['valuePI'], -kwargs['valuePD'], kwargs['valueEnabled'])
        else:
            agent.setControllerParams(name, kwargs['valueP'], kwargs['valuePI'], kwargs['valuePD'], kwargs['valueEnabled'])
        #change setpoint
        env.sim.jsbsim[propName] = kwargs['valueSetPoint']

    return tuningCallback
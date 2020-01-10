#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import gym
import numpy as np

import gym_jsbsim

from gym_jsbsim.wrappers import PidWrapper, PidWrapperParams, PidParameters

ENV_ID = "JSBSim-SteadyRollAngleTask-Cessna172P-Shaping.STANDARD-FG-v0"

env = gym.make(ENV_ID)
# elevator params: 'Kp':  -5e-2, 'Ki': -6.5e-2, 'Kd': -1e-3
# aileron prams:   'Kp': 3.5e-2, 'Ki':    1e-2, 'Kd': 0.0
elevator_wrap = PidWrapperParams('fcs_elevator_cmd_norm', 'error_glideAngle_error_deg', PidParameters( -5e-2, -6.5e-2, -1e-3))
aileron_wrap  = PidWrapperParams('fcs_aileron_cmd_norm',  'error_rollAngle_error_deg',  PidParameters(3.5e-2,    1e-2,   0.0))
# env = PidWrapper(env, [aileron_wrap, elevator_wrap])
env = PidWrapper(env, [aileron_wrap])

state = env.reset()
obs = env.step([666])

print("initial state: {}".format(state))

action_space = env.action_space

print(action_space)
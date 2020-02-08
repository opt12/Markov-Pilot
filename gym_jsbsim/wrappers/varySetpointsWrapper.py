#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import gym
import numpy as np
import random
import gym_jsbsim.properties as prp


class VarySetpointsWrapper(gym.Wrapper):
    """
    A wrapper to vary the setpoints at the beginning of each episode

    This can be used during training to have bigger vriance in the training data
    """
    
    def __init__(self, env):
        super(VarySetpointsWrapper, self).__init__(env)
        self.original_env = env
    
    def step(self, action):
        return self.original_env.step(action)

    def reset(self):
        tgt_flight_path_deg = random.uniform(-12, -5.5)
        tgt_roll_angle_deg  = random.uniform(-15, -5)
        self.original_env.task.change_setpoints(self.original_env.sim, { 
            prp.setpoint_flight_path_deg: tgt_flight_path_deg
          , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg 
          })
        
        return self.original_env.reset()


    




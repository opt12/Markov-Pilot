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

    This can be used during training to have bigger variance in the training data
    """
    
    def __init__(self, env, modulation_amplitude = None, modulation_period = 120):
        super(VarySetpointsWrapper, self).__init__(env)
        self.original_env = env
        self.step_width = 2 * np.pi / modulation_period
        self.modulation_amplitude = modulation_amplitude
        if self.modulation_amplitude:
            self.roll_modulation_amplitude = 0
        self.step_counter = 0
    
    def step(self, action):
        if self.modulation_amplitude:
            modulation = np.sin(self.step_counter * self.step_width) * self.roll_modulation_amplitude
            self.original_env.task.change_setpoints(self.original_env.sim, { 
                    prp.setpoint_roll_angle_deg:  self.tgt_roll_angle_deg + modulation
            })
        self.step_counter += 1
        return self.original_env.step(action)

    def reset(self):
        tgt_flight_path_deg = random.uniform(-12, -5.5)
        self.tgt_roll_angle_deg  = random.uniform(-15, 15)
        if self.modulation_amplitude:
            self.roll_modulation_amplitude = self.modulation_amplitude * self.tgt_roll_angle_deg
        else:
            self.roll_modulation_amplitude = 0
        # initial_path_angle_gamma_deg = tgt_flight_path_deg
        # initial_roll_angle_phi_deg   = tgt_roll_angle_deg
        initial_fwd_speed_KAS        = random.uniform(65, 110)

        self.original_env.task.change_setpoints(self.original_env.sim, { 
            prp.setpoint_flight_path_deg: tgt_flight_path_deg
          , prp.setpoint_roll_angle_deg:  self.tgt_roll_angle_deg 
          })
        self.original_env.task.set_initial_ac_attitude( {  prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                    #    , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                    #    , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                       #, prp.initial_aoa_deg: initial_aoa_deg
                                      })
        
        return self.original_env.reset()


    




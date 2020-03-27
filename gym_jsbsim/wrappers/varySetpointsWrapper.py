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
    
    def __init__(self, env, modulation_amplitude = None, modulation_period = 120, modulation_decay = 1):
        super(VarySetpointsWrapper, self).__init__(env)
        self.env = env
        self.step_width = 2 * np.pi / modulation_period
        self.modulation_amplitude = modulation_amplitude / modulation_decay if modulation_amplitude else None
        self.modulation_decay = modulation_decay
        # if self.modulation_amplitude:
        #     self.roll_modulation_amplitude = 0
        self.step_counter = 0
    
    def step(self, action):
        if self.modulation_amplitude:
            modulation = np.sin(self.step_counter * self.step_width) * self.modulation_amplitude
            self.env.task.change_setpoints(self.env.sim, { 
                    prp.setpoint_roll_angle_deg:  self.tgt_roll_angle_deg + modulation
            })
        self.step_counter += 1
        return self.env.step(action)

    def reset(self):
        if not self.modulation_amplitude:
            tgt_flight_path_deg = random.uniform(-12, -5.5)
            self.tgt_roll_angle_deg  = random.uniform(-15, 15)  #object variable to support roll modulation

            initial_path_angle_gamma_deg = random.uniform(-12, -5.5)
            initial_roll_angle_phi_deg   = random.uniform(-15, 15)
            initial_fwd_speed_KAS        = random.uniform(65, 110)
        else:
            tgt_flight_path_deg = self.env.sim[prp.setpoint_flight_path_deg]
            self.tgt_roll_angle_deg  = self.env.sim[prp.setpoint_roll_angle_deg]

            initial_path_angle_gamma_deg = self.env.sim[prp.initial_flight_path_deg]
            initial_roll_angle_phi_deg   = self.env.sim[prp.initial_roll_deg]
            initial_fwd_speed_KAS        = self.env.sim[prp.initial_u_fps]/1.6878099110965
            self.modulation_amplitude *= self.modulation_decay
            print(f"sine wave modulation amplitude set to ±{self.modulation_amplitude}")

        self.env.task.change_setpoints(self.env.sim, { 
            prp.setpoint_flight_path_deg: tgt_flight_path_deg
          , prp.setpoint_roll_angle_deg:  self.tgt_roll_angle_deg 
          })
        self.env.task.set_initial_ac_attitude( {  prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                        , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                        , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                        # , prp.initial_aoa_deg: initial_aoa_deg
                                      })
        
        return self.env.reset()
    
    def set_modulation_params(self, modulation_amplitude = None, modulation_period = None, modulation_decay = None):
        if modulation_period:
            self.step_width = 2 * np.pi / modulation_period
        if modulation_decay:
            self.modulation_decay = modulation_decay
        if modulation_amplitude:
            self.modulation_amplitude = modulation_amplitude / self.modulation_decay
        else:
            self.modulation_amplitude = None
        



    




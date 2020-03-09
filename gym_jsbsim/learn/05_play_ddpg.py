#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import argparse
import gym
import pybullet_envs

from lib import model

import numpy as np
import torch

import gym_jsbsim
from gym_jsbsim.wrappers import EpisodePlotterWrapper, PidWrapper, PidWrapperParams, PidParameters, StateSelectWrapper
import gym_jsbsim.properties as prp


ENV_ID = "JSBSim-SteadyRollGlideTask-Cessna172P-Shaping.STANDARD-NoFG-v0"
ENV_ID = "JSBSim-SteadyRollAngleTask-Cessna172P-Shaping.STANDARD-NoFG-v0"
# ENV_ID = "JSBSim-SteadyRollAngleTask-Cessna172P-Shaping.EXTRA-NoFG-v0"
PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    # parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    parser.add_argument("--repeat", default= 1, type=int, help="The number of repetitions to be played")
    args = parser.parse_args()

    # elevator params: 'Kp':  -5e-2, 'Ki': -6.5e-2, 'Kd': -1e-3
    # aileron prams:   'Kp': 3.5e-2, 'Ki':    1e-2, 'Kd': 0.0
    elevator_wrap = PidWrapperParams('fcs_elevator_cmd_norm', 'error_glideAngle_error_deg', PidParameters( -5e-2, -6.5e-2, -1e-3))
    aileron_wrap  = PidWrapperParams('fcs_aileron_cmd_norm',  'error_rollAngle_error_deg',  PidParameters(3.5e-2,    1e-2,   0.0))

    env = gym.make(ENV_ID)
    env = EpisodePlotterWrapper(env)    #to show a summary of the next epsode, set env.showNextPlot(True)
    env = PidWrapper(env, [elevator_wrap])  #to apply PID control to the pitch axis
    # env = PidWrapper(env, [elevator_wrap, aileron_wrap])  #to apply PID control to the pitch and the roll axis (for benchmarking) #remove net.load_state_dict() !!!
    # env = StateSelectWrapper(env, ['error_rollAngle_error_deg', 'velocities_p_rad_sec'])#, 'attitude_roll_rad', 'velocities_p_rad_sec'])
    env = StateSelectWrapper(env, PRESENTED_STATE)



    # if args.record:
    #     env = gym.wrappers.Monitor(env, args.record)

    net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model))

    tgt_flight_path_deg = -7.5
    tgt_roll_angle_deg  = -10
    episode_steps   = 2500
    initial_fwd_speed_KAS        = 75
    initial_path_angle_gamma_deg = -0
    initial_roll_angle_phi_deg   = -0
    initial_aoa_deg              = 1.0

    for i in range(args.repeat):
        # put all the value into variables, so I can manipulate them conveniently in the debugger
        env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                           , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg
                                           , prp.episode_steps:            episode_steps})
        env.task.set_initial_ac_attitude( {  prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                           , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                           , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                           , prp.initial_aoa_deg: initial_aoa_deg
                                           })

        obs = env.reset()
        total_reward = 0.0
        total_steps = 0
        env.showNextPlot(True, False, True)   #show the plot, but don't save the png
        while True:
            obs_v = torch.FloatTensor([obs])
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            # env.render()
            total_reward += reward
            total_steps += 1
            if total_steps % 500 == 0:
                tgt_roll_angle_deg = -tgt_roll_angle_deg
                env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                                   , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg  })
            if done:
                break
        print("In %d steps we got %.3f reward" % (total_steps, total_reward))

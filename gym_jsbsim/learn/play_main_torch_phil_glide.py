#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import time
import random

from ddpg_torch import Agent
import gym
import numpy as np

import gym_jsbsim
from gym_jsbsim.wrappers import EpisodePlotterWrapper, PidWrapper, PidWrapperParams, PidParameters, StateSelectWrapper, VarySetpointsWrapper
import gym_jsbsim.properties as prp


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    # args = parser.parse_args()
    # device = torch.device("cuda" if args.cuda else "cpu")

    ENV_ID = "JSBSim-SteadyGlideAngleTask-Cessna172P-Shaping.STANDARD-NoFG-v0"
    CHKPT_DIR = ENV_ID + "MovementPunishment"
    CHKPT_POSTFIX = ""
    SAVED_MODEL_NAME = "glide_best"
    SAVED_MODEL_NAME = "glide_+509.919_599"


    GAMMA = .95
    BATCH_SIZE = 64
    LEARNING_RATE_ACTOR = 1e-4
    LEARNING_RATE_CRITIC = 1e-3
    REPLAY_SIZE = 1000000
    TEST_ITERS = 2000
    INTERACTION_FREQ = 5
    # PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec']
    PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec', 'error_rollAngle_error_integral_deg_sec',  
                       'error_glideAngle_error_deg', 'velocities_q_rad_sec', 'error_glideAngle_error_integral_deg_sec',
                       'velocities_vc_kts', 
                       'info_delta_cmd_aileron', 'fcs_aileron_cmd_norm', 
                       'info_delta_cmd_elevator', 'fcs_elevator_cmd_norm']
# PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec', 'info_delta_cmd_aileron', 'fcs_aileron_cmd_norm']
    # PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec', 'velocities_vc_kts']

    # save_path = os.path.join("saves", "{}_ddpg-gamma0_95-two-state_5Hz_alpha_5e-5_beta_5e-4_100x100_size".format(datetime.datetime.now().strftime("%Y_%m_%d-%H:%M")) + args.name)
    # os.makedirs(save_path, exist_ok=True)

    # elevator params: 'Kp':  -5e-2, 'Ki': -6.5e-2, 'Kd': -1e-3
    # aileron prams:   'Kp': 3.5e-2, 'Ki':    1e-2, 'Kd': 0.0
    elevator_wrap = PidWrapperParams('fcs_elevator_cmd_norm', 'error_glideAngle_error_deg', PidParameters( -5e-2, -6.5e-2, -1e-3))
    aileron_wrap  = PidWrapperParams('fcs_aileron_cmd_norm',  'error_rollAngle_error_deg',  PidParameters(3.5e-2,    1e-2,   0.0))

    env = gym.make(ENV_ID, agent_interaction_freq = INTERACTION_FREQ)
    env = VarySetpointsWrapper(env, modulation_amplitude = None, modulation_period = 150, modulation_decay=0.99)     #to vary the setpoints during training
    env = EpisodePlotterWrapper(env, presented_state=PRESENTED_STATE)    #to show a summary of the next epsode, set env.showNextPlot(True)
    env = PidWrapper(env, [aileron_wrap])  #to apply PID control to the pitch axis
    env = StateSelectWrapper(env, PRESENTED_STATE )
    # env = StateSelectWrapper(env, ['error_glideAngle_error_deg', 'velocities_r_rad_sec'])#, 'attitude_roll_rad', 'velocities_p_rad_sec'])
    print("env.observation_space: {}".format(env.observation_space))

    tgt_flight_path_deg = -6.5
    tgt_roll_angle_deg  = -5
    episode_steps   = 2500  #2*60*INTERACTION_FREQ
    initial_fwd_speed_KAS        = 130
    initial_path_angle_gamma_deg = -0
    initial_roll_angle_phi_deg   = -0
    initial_aoa_deg              = 1.0

    env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                        , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg
                                        , prp.episode_steps:            episode_steps})
    env.task.set_initial_ac_attitude( {  prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                       , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                       , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                       , prp.initial_aoa_deg: initial_aoa_deg
                                      })
    # TODO: a of this stuff is unnecessary, but #I need an agent right now.
    play_agent = Agent(env=env, input_dims = [env.observation_space.shape[0]], n_actions = env.action_space.shape[0],
              lr_actor=LEARNING_RATE_ACTOR, lr_critic=LEARNING_RATE_CRITIC, tau=0.001,  #irrelevant for pure playing
              layer1_size=400, layer2_size=300, 
              chkpt_dir=CHKPT_DIR, chkpt_postfix=CHKPT_POSTFIX)  #TODO: pass summary writer to Agent
    
    play_agent.load_models(name_discriminator = SAVED_MODEL_NAME)

    np.random.seed(0)

    obs = env.reset()
    env.showNextPlot(True, True)
    done = False
    score = 0
    # train_agent.reset_noise_source()    #this is like in the original paper
    total_steps = 0
    ts = time.time()
    while not done:
        act = play_agent.choose_action(obs, add_exploration_noise=False)
        new_state, reward, done, info = env.step(act)
        score += reward     # the action includes noise!!!
        obs = new_state
        #env.render()
        total_steps += 1
        if total_steps % 350 == 0:
            tgt_roll_angle_deg = -tgt_roll_angle_deg
            tgt_flight_path_deg = random.uniform(-12, -5.5)
            env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                                , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg  })
    delta_t = time.time() - ts

    print('episode finished; score %.2f;' % score,
            '%d' % total_steps, 'steps in %.2f' % delta_t, 'sec; That\'s %.2f' % (total_steps/delta_t),'steps/sec')


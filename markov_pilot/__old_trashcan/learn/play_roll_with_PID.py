#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import time
import random


from ddpg_torch import Agent
import gym
import numpy as np

import gym_jsbsim
from markov_pilot.wrappers import EpisodePlotterWrapper, PidWrapper, PidWrapperParams, PidParameters, StateSelectWrapper, VarySetpointsWrapper
import markov_pilot.properties as prp

ENV_ID = "JSBSim-SteadyRollAngleTask-Cessna172P-Shaping.STANDARD-FG-v0"
CHKPT_DIR = ENV_ID
CHKPT_DIR = "JSBSim-SteadyRollAngleTask-Cessna172P-Shaping.STANDARD-NoFG-v0"  #use this if you want to perform Flightgear Rendering
CHKPT_DIR = CHKPT_DIR + "avoid_overshoot"
# CHKPT_POSTFIX = "travel_error"
CHKPT_POSTFIX = "angular_veocity"     
#it looks like the angular-velocity criterion is more helpful to avoid flittering. 
#The control surface travel (derivative) must be presented to the ANN anyways.
SAVED_MODEL_DISCRIMINATOR = "roll_best"
# SAVED_MODEL_DISCRIMINATOR = "roll_+584.69"
ENABLE_PARALLEL_PID = 0

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    # args = parser.parse_args()

    #all of this stuff is not really necessary to play a model, but needed to build the name
    GAMMA = .95
    BATCH_SIZE = 64
    LEARNING_RATE_ACTOR = 1e-4
    LEARNING_RATE_CRITIC = 1e-3
    REPLAY_SIZE = 1000000
    TEST_ITERS = 2000
    INTERACTION_FREQ = 5

    PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec', 'error_rollAngle_error_integral_deg_sec',  
                       'error_glideAngle_error_deg', 'velocities_q_rad_sec', 'error_glideAngle_error_integral_deg_sec',
                       'velocities_vc_kts', 
                       'info_delta_cmd_aileron', 'fcs_aileron_cmd_norm', 
                       'info_delta_cmd_elevator', 'fcs_elevator_cmd_norm']

    # elevator params: 'Kp':  -5e-2, 'Ki': -6.5e-2, 'Kd': -1e-3
    # aileron prams:   'Kp': 3.5e-2, 'Ki':    1e-2, 'Kd': 0.0
    elevator_wrap = PidWrapperParams('fcs_elevator_cmd_norm', 'error_glideAngle_error_deg', PidParameters( -5e-2, -6.5e-2, -1e-3))
    aileron_wrap  = PidWrapperParams('fcs_aileron_cmd_norm',  'error_rollAngle_error_deg',  PidParameters(3.5e-2,    1e-2,   0.0))

    env = gym.make(ENV_ID, agent_interaction_freq = INTERACTION_FREQ)
    env = VarySetpointsWrapper(env, modulation_amplitude = 10, modulation_period = 50)     #to vary the setpoints during training
    env = EpisodePlotterWrapper(env, presented_state=PRESENTED_STATE)    #to show a summary of the next epsode, set env.showNextPlot(True)
    env = PidWrapper(env, [elevator_wrap])  #to apply PID control to the pitch axis
    env = StateSelectWrapper(env, PRESENTED_STATE )

    if ENABLE_PARALLEL_PID:
        env_pid = gym.make(ENV_ID, agent_interaction_freq = INTERACTION_FREQ)
        # env_pid = VarySetpointsWrapper(env_pid, modulation_amplitude = 10, modulation_period = 50)     #to vary the setpoints during training
        env_pid = EpisodePlotterWrapper(env_pid, presented_state=PRESENTED_STATE)    #to show a summary of the next epsode, set env.showNextPlot(True)
        env_pid = PidWrapper(env_pid, [elevator_wrap, aileron_wrap])  #to apply PID control to the pitch axis
        env_pid = StateSelectWrapper(env_pid, PRESENTED_STATE )

    tgt_flight_path_deg = -8.5
    tgt_roll_angle_deg  = -20
    episode_steps   = 2500  #2*60*INTERACTION_FREQ
    initial_fwd_speed_KAS        = 100
    initial_path_angle_gamma_deg = -8.5
    initial_roll_angle_phi_deg   = -20
    initial_aoa_deg              = 1.0

    env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                        , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg
                                        , prp.episode_steps:            episode_steps})
    env.task.set_initial_ac_attitude( {  prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                       , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                       , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                       , prp.initial_aoa_deg: initial_aoa_deg
                                      })
    if ENABLE_PARALLEL_PID:
        env_pid.task.change_setpoints(env_pid.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                        , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg
                                        , prp.episode_steps:            episode_steps})
        env_pid.task.set_initial_ac_attitude( {  prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                       , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                       , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                       , prp.initial_aoa_deg: initial_aoa_deg
                                      })
    # TODO: a of this stuff is unnecessary, but I need an agent right now.
    play_agent = Agent(lr_actor=LEARNING_RATE_ACTOR, lr_critic=LEARNING_RATE_CRITIC, input_dims = [env.observation_space.shape[0]], tau=0.001, env=env,
              batch_size=BATCH_SIZE,  layer1_size=400, layer2_size=300, n_actions = env.action_space.shape[0],
              chkpt_dir=CHKPT_DIR, chkpt_postfix=CHKPT_POSTFIX )  #TODO: action space should be env.action_space.shape[0]
    
    env.set_meta_information(env_info = 'ANN roll control Player')
    env.set_meta_information(model_discriminator = SAVED_MODEL_DISCRIMINATOR)

    if ENABLE_PARALLEL_PID:
        env_pid.set_meta_information(env_info = 'PID roll control Player')
        env_pid.set_meta_information(model_discriminator = 'full PID control')
        env_pid.set_meta_information(model_type = 'PID')
    
    play_agent.load_models(name_discriminator = SAVED_MODEL_DISCRIMINATOR)


    np.random.seed(0)

    obs_ann = env.reset()
    env.showNextPlot(True, True)
    score_ann = 0

    if ENABLE_PARALLEL_PID:
        env_pid.showNextPlot(True, True)
        _ = env_pid.reset()
        score_pid = 0
    
    done = False
    done_pid = False

    total_steps = 0
    ts = time.time()
    while not done:
        act = play_agent.choose_action(obs_ann, add_exploration_noise=False)    #no noise when testing
        obs_ann, reward_ann, done, info = env.step(act)
        score_ann += reward_ann

        if ENABLE_PARALLEL_PID:
            _, reward_pid, done_pid, _ = env_pid.step([])
            score_pid += reward_pid

        done = done or done_pid #stop episode if either of the controllers is done

        # env.render('flightgear')  #when rendering in Flightgear, the environment must be changed as well
        total_steps += 1
        if total_steps % (20*INTERACTION_FREQ) == 0:
            tgt_roll_angle_deg = random.uniform(-30, 30)
            env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                                , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg  })
            if ENABLE_PARALLEL_PID:
                env_pid.task.change_setpoints(env_pid.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                                , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg  })
        if total_steps % (25*INTERACTION_FREQ) == 0:
            tgt_flight_path_deg = random.uniform(-5.5, -12.5)
            env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg  })
            if ENABLE_PARALLEL_PID:
                env_pid.task.change_setpoints(env_pid.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg })
    delta_t = time.time() - ts

    print('ANN: episode finished; score %.2f;' % score_ann,
            '%d' % total_steps, 'steps in %.2f' % delta_t, 'sec; That\'s %.2f' % (total_steps/delta_t),'steps/sec')
    env.close()
    
    if ENABLE_PARALLEL_PID:
        print('PID: episode finished; score %.2f;' % score_pid,
                '%d' % total_steps, 'steps in %.2f' % delta_t, 'sec; That\'s %.2f' % (total_steps/delta_t),'steps/sec')
        env_pid.close()            



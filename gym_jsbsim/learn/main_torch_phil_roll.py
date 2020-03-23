#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import time

from ddpg_torch import Agent
import gym
import numpy as np

import gym_jsbsim
from gym_jsbsim.wrappers import EpisodePlotterWrapper, PidWrapper, PidWrapperParams, PidParameters, StateSelectWrapper, VarySetpointsWrapper
import gym_jsbsim.properties as prp


best_reward = None  #we don't like globals, but it really helps here

def test_net(agent, env, add_exploration_noise=False):
    global best_reward
    exploration_noise = add_exploration_noise   #to have a handle on that in the debugger
    obs = env.reset()
    env.showNextPlot(True, True)
    done = False
    score = 0
    steps = 0
    while not done:
        act = agent.choose_action(obs, add_exploration_noise=exploration_noise)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))  #TODO: is it a good idea to remeber the test episodes? Why not?
        score += reward     # the action includes noise!!!
        obs = new_state
        steps += 1
        #env.render()
    print("\tTest yielded a score of %.2f" %score, ".")

    name = "roll_%+.3f_%d" % (score, steps)
    agent.save_models(name_discriminator=name)    
    if best_reward is None or best_reward < score:
        if best_reward is not None:
            print("Best reward updated: %.3f -> %.3f" % (best_reward, score))
        name = "roll_%+.3f_%d" % (score, steps)
        agent.save_models(name_discriminator=name)
        agent.save_models(name_discriminator='roll_best')
        best_reward = score


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    # args = parser.parse_args()
    # device = torch.device("cuda" if args.cuda else "cpu")

    ENV_ID = "JSBSim-SteadyRollAngleTask-Cessna172P-Shaping.STANDARD-NoFG-v0"
    CHKPT_DIR = ENV_ID + "MovementPunishment"
    CHKPT_POSTFIX = "bigger_state_presentation"

    GAMMA = .95
    BATCH_SIZE = 64
    LEARNING_RATE_ACTOR = 1e-4
    LEARNING_RATE_CRITIC = 1e-3
    REPLAY_SIZE = 1000000
    TEST_ITERS = 2000
    INTERACTION_FREQ = 5
    # PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec']
    PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec', 'info_delta_cmd_aileron', 'fcs_aileron_cmd_norm', 'velocities_vc_kts', 'error_rollAngle_error_integral_deg_sec']
    PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec', 'velocities_vc_kts', 'error_rollAngle_error_integral_deg_sec']
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
    env = PidWrapper(env, [elevator_wrap])  #to apply PID control to the pitch axis
    env = StateSelectWrapper(env, PRESENTED_STATE )
    print("env.observation_space: {}".format(env.observation_space))

    tgt_flight_path_deg = -10
    tgt_roll_angle_deg  = 10
    episode_steps   = 2*60*INTERACTION_FREQ
    initial_path_angle_gamma_deg = 0
    initial_roll_angle_phi_deg   = 0
    initial_fwd_speed_KAS        = 95
    initial_aoa_deg              = 1.0
    env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                        , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg
                                        , prp.episode_steps:            episode_steps})
    env.task.set_initial_ac_attitude( {  prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                       , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                       , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                       , prp.initial_aoa_deg: initial_aoa_deg
                                      })

    test_env = gym.make(ENV_ID,  agent_interaction_freq = INTERACTION_FREQ)
    test_env = VarySetpointsWrapper(test_env, modulation_amplitude = None, modulation_period = 150)     #to vary the setpoints during training
    test_env = EpisodePlotterWrapper(test_env, presented_state=PRESENTED_STATE)    #to show a summary of the next epsode, set env.showNextPlot(True)
    test_env = PidWrapper(test_env, [elevator_wrap]) #to apply PID control to the pitch axis
    test_env = StateSelectWrapper(test_env, PRESENTED_STATE)
    # test_env = StateSelectWrapper(test_env, ['error_glideAngle_error_deg', 'velocities_r_rad_sec'])#, 'attitude_roll_rad', 'velocities_p_rad_sec'])

    test_env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                        , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg
                                        , prp.episode_steps:            episode_steps})
    test_env.task.set_initial_ac_attitude( {  prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                       , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                       , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                       , prp.initial_aoa_deg: initial_aoa_deg
                                      })
    #TODO: open summary writer here

    train_agent = Agent(lr_actor=LEARNING_RATE_ACTOR, lr_critic=LEARNING_RATE_CRITIC, input_dims = [env.observation_space.shape[0]], tau=0.001, env=env,
              batch_size=BATCH_SIZE,  layer1_size=400, layer2_size=300, n_actions = env.action_space.shape[0],
              chkpt_dir=CHKPT_DIR, chkpt_postfix=CHKPT_POSTFIX)  #TODO: pass summary writer to Agent

    np.random.seed(0)

    score_history = []

    exploration_noise_flag = True

    for episode in range(1000):
        obs = env.reset()
        done = False
        score = 0
        # train_agent.reset_noise_source()    #this is like in the original paper
        train_agent.reduce_noise_sigma(sigma_factor=0.98)
        steps = 0
        ts = time.time()
        while not done:
            act = train_agent.choose_action(obs, add_exploration_noise = exploration_noise_flag)
            new_state, reward, done, info = env.step(act)
            train_agent.remember(obs, act, reward, new_state, int(done))
            train_agent.learn()
            score += reward     # the action includes noise!!!
            obs = new_state
            #env.render()
            steps += 1
        score_history.append(score)
        delta_t = time.time() - ts

        print('episode ', episode, 'score %.2f;' % score,
              '%d' % steps, 'steps in %.2f' % delta_t, 'sec; That\'s %.2f' % (steps/delta_t),'steps/sec', 
            'trailing 15 games avg %.3f' % np.mean(score_history[-15:]))

        if episode% 5 == 0:
            test_net(train_agent, test_env, add_exploration_noise=False)
        
        # if episode == 40:   #switch to sine wave exploration after 30 "normal" episodes
        #     exploration_noise_flag = False
        #     env.set_modulation_params(modulation_amplitude = 0.7)
            

        # if i % 25 == 0:
        #     train_agent.save_models()


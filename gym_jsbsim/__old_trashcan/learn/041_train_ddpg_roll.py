#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import os
import ptan
import time
import datetime
import gym
import argparse
from tensorboardX import SummaryWriter
import numpy as np
import math

from lib import model, common

import torch
import torch.optim as optim
import torch.nn.functional as F

import gym_jsbsim
from gym_jsbsim.wrappers import EpisodePlotterWrapper, PidWrapper, PidWrapperParams, PidParameters, StateSelectWrapper, VarySetpointsWrapper
import gym_jsbsim.properties as prp


# ENV_ID = "Pendulum-v0"
# ENV_ID = "JSBSim-SteadyRollAngleTask-Cessna172P-Shaping.STANDARD-FG-v0"

def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    render = False
    for i in range(count):
        obs = env.reset()
        if i == count -1:
            env.showNextPlot(True, True)
        render = False
        while True:
            if render:
                env.render()
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    ENV_ID = "JSBSim-SteadyRollAngleTask-Cessna172P-Shaping.STANDARD-NoFG-v0"

    GAMMA = .95
    BATCH_SIZE = 128
    LEARNING_RATE_ACTOR = 1e-4
    LEARNING_RATE_CRITIC = 1e-4
    LEARNING_RATE_ACTOR = 0.00005
    LEARNING_RATE_CRITIC = 0.0005
    # alpha=0.00005# LR actor
    # beta=0.0005# Lr critic
    REPLAY_SIZE = 100000
    REPLAY_INITIAL = 10000
    TEST_ITERS = 2000
    INTERACTION_FREQ = 5
    PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec']
    # PRESENTED_STATE = ['error_rollAngle_error_deg', 'velocities_p_rad_sec']'info_delta_cmd_aileron', 'fcs_aileron_cmd_norm']

    save_path = os.path.join("saves", "{}_ddpg-gamma0_95-two-state_5Hz_alpha_5e-5_beta_5e-4_100x100_size".format(datetime.datetime.now().strftime("%Y_%m_%d-%H-%M")) + args.name)
    os.makedirs(save_path, exist_ok=True)

    # elevator params: 'Kp':  -5e-2, 'Ki': -6.5e-2, 'Kd': -1e-3
    # aileron prams:   'Kp': 3.5e-2, 'Ki':    1e-2, 'Kd': 0.0
    elevator_wrap = PidWrapperParams('fcs_elevator_cmd_norm', 'error_glideAngle_error_deg', PidParameters( -5e-2, -6.5e-2, -1e-3))
    aileron_wrap  = PidWrapperParams('fcs_aileron_cmd_norm',  'error_rollAngle_error_deg',  PidParameters(3.5e-2,    1e-2,   0.0))

    env = gym.make(ENV_ID, agent_interaction_freq = INTERACTION_FREQ)
    env = VarySetpointsWrapper(env)     #to vary the setpoints during training
    env = EpisodePlotterWrapper(env)    #to show a summary of the next epsode, set env.showNextPlot(True)
    env = PidWrapper(env, [elevator_wrap])  #to apply PID control to the pitch axis
    env = StateSelectWrapper(env, PRESENTED_STATE )
    # env = PidWrapper(env, [aileron_wrap])  #to apply PID control to the pitch axis
    # env = StateSelectWrapper(env, ['error_glideAngle_error_deg', 'velocities_r_rad_sec'])#, 'attitude_roll_rad', 'velocities_p_rad_sec'])
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
    #test_env = VarySetpointsWrapper(test_env)     #to vary the setpoints during training
    test_env = EpisodePlotterWrapper(test_env)    #to show a summary of the next epsode, set env.showNextPlot(True)
    test_env = PidWrapper(test_env, [elevator_wrap]) #to apply PID control to the pitch axis
    test_env = StateSelectWrapper(test_env, PRESENTED_STATE)
    # test_env = PidWrapper(test_env, [aileron_wrap]) #to apply PID control to the pitch axis
    # test_env = StateSelectWrapper(test_env, ['error_glideAngle_error_deg', 'velocities_r_rad_sec'])#, 'attitude_roll_rad', 'velocities_p_rad_sec'])

    test_env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                        , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg
                                        , prp.episode_steps:            episode_steps})
    test_env.task.set_initial_ac_attitude( {  prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                       , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                       , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                       , prp.initial_aoa_deg: initial_aoa_deg
                                      })
                                      
    act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = model.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_" + ENV_ID + args.name)
    #TODO: Check the OU-parameters! It looks like with standard prams, the noise is far too heavy.
    agent = model.AgentDDPG(act_net, device=device)#, ou_mu=0.0, ou_teta=0.5, ou_sigma=0.05, ou_epsilon=1.0)
    # agent = model.PidWithDDPG(act_net, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE_ACTOR)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE_CRITIC)

    frame_idx = 0
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = common.unpack_batch_ddqn(batch, device)
                if math.isnan(states_v.sum().item()) or \
                   math.isnan(actions_v.sum().item()) or \
                   math.isnan(rewards_v.sum().item()) or \
                   math.isnan(dones_mask.sum().item()):
                    print (last_states_v.numpy()[0])

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                        name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(act_net.state_dict(), fname)
                        best_reward = rewards
                    else:
                        #save the latest model with every test
                        name = "latest.dat"
                        fname = os.path.join(save_path, name)
                        torch.save(act_net.state_dict(), fname)



    pass

#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import markov_pilot.properties as prp

best_reward = None  #we don't like globals, but it really helps here

def test_net(agent, env, add_exploration_noise=False):
    global best_reward

    tgt_flight_path_deg = -6.5
    tgt_roll_angle_deg  = 10
    episode_steps   = 2*60* env.task.step_frequency_hz
    initial_path_angle_gamma_deg = 0
    initial_roll_angle_phi_deg   = 0
    initial_fwd_speed_KAS        = 75
    initial_aoa_deg              = 1.0

    env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg
                                        , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg
                                        , prp.episode_steps:            episode_steps})
    env.task.set_initial_ac_attitude( {  prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                       , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                       , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                       , prp.initial_aoa_deg: initial_aoa_deg
                                      })

    exploration_noise = add_exploration_noise   #to have a handle on that in the debugger
    obs = env.reset()
    env.showNextPlot(True, True)
    done = False
    score = 0
    steps = 0
    while not done:
        if agent:
            act = agent.choose_action(obs, add_exploration_noise=exploration_noise)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))  #TODO: is it a good idea to remeber the test episodes? Why not?
        else:   #we have a fully wrapped env
            new_state, reward, done, info = env.step([])
        score += reward     # the action includes noise!!!
        obs = new_state
        steps += 1
        if steps == int(0.5 *60* env.task.step_frequency_hz):
            tgt_flight_path_deg = -6.5
            tgt_roll_angle_deg  = -10
            env.task.change_setpoints(env.sim, { prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg })
        if steps == int(1 *60* env.task.step_frequency_hz):
            tgt_flight_path_deg = -7.5
            tgt_roll_angle_deg  = -10
            env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: tgt_flight_path_deg })
        if steps == int(1.5 *60* env.task.step_frequency_hz):
            tgt_flight_path_deg = -7.5
            tgt_roll_angle_deg  = 10
            env.task.change_setpoints(env.sim, { prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg })

        #env.render()
    print("\tTest yielded a score of %.2f" %score, ".")

    name = env.meta_dict['model_base_name']
    discriminator = env.meta_dict['model_discriminator']
    env.set_meta_information(model_discriminator = discriminator)
    agent.save_models(name_discriminator=discriminator)    
    if best_reward is None or best_reward < score:
        if best_reward is not None:
            print("Best reward updated: %.3f -> %.3f" % (best_reward, score))
        agent.save_models(name_discriminator= name + '_best')
        best_reward = score


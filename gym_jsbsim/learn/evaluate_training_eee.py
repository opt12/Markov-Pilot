#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import numpy as np
import gym_jsbsim.properties as prp

best_score_n = []  #we don't like globals, but it really helps here

def test_net(agents, env, add_exploration_noise=False):
    global best_score_n 
    if len(best_score_n) != env.n:
        best_score_n = np.zeros(env.n)

    tgt_flight_path_deg = -6.5
    tgt_roll_angle_deg  = 10
    initial_path_angle_gamma_deg = 0
    initial_roll_angle_phi_deg   = 0
    initial_fwd_speed_KAS        = 75
    initial_aoa_deg              = 1.0

    env.change_setpoints({ prp.setpoint_flight_path_deg: tgt_flight_path_deg
                         , prp.setpoint_roll_angle_deg:  tgt_roll_angle_deg
                        })
    env.set_initial_conditions( { prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                , prp.initial_aoa_deg: initial_aoa_deg
                                })

    exploration_noise = add_exploration_noise   #to have a handle on that in the debugger
    obs_n = env.reset()
    env.showNextPlot(True, True)
    terminal = False
    score_n = np.zeros(env.n)
    steps = 0
    while not terminal:
        # get action
        action_n = [agent.action(obs, exploration_noise) for agent, obs in zip(agents,obs_n)]

        new_state_n, reward_n, done_n, info_n = env.step(action_n)

        # agent.remember(obs, act, reward, new_state, int(terminal))  #TODO: is it a good idea to remember the test episodes? Why not?

        score_n += reward_n     # the action includes noise!!!
        obs_n = new_state_n
        steps += 1
        if steps == int(0.5 *60 / env.dt):
            tgt_flight_path_deg = -6.5
            tgt_roll_angle_deg  = -10
            env.change_setpoints({ prp.roll_deg:  tgt_roll_angle_deg })
        if steps == int(1 *60 / env.dt):
            tgt_flight_path_deg = -7.5
            tgt_roll_angle_deg  = -10
            env.change_setpoints({ prp.flight_path_deg: tgt_flight_path_deg })
        if steps == int(1.5 *60 / env.dt):
            tgt_flight_path_deg = -7.5
            tgt_roll_angle_deg  = 10
            env.change_setpoints({ prp.roll_deg:  tgt_roll_angle_deg })

        terminal = any(done_n) or env.is_terminal()
        #env.render()
    print("\tTest yielded a score of : [", end="")
    print(*["%.2f"%sc for sc in score_n], sep = ", ", end="")
    print("].")

    for idx, score in enumerate(score_n):
        if best_score_n[idx] < score:
            print("%s: Best score updated: %.3f -> %.3f" % (agents[idx].name, best_score_n[idx], score))
            # agent.save_models(name_discriminator= name + '_best')
            best_score_n[idx] = score


    # name = env.meta_dict['model_base_name']
    # discriminator = env.meta_dict['model_discriminator']
    # env.set_meta_information(model_discriminator = discriminator)
    # agent.save_models(name_discriminator=discriminator)    
    # if best_reward is None or best_reward < score:
    #     if best_reward is not None:
    #         print("Best reward updated: %.3f -> %.3f" % (best_reward, score))
    #     agent.save_models(name_discriminator= name + '_best')
    #     best_reward = score

if __name__ == '__main__':

    from gym_jsbsim.agent_task_eee import SingleChannel_FlightAgentTask
    from gym_jsbsim.agents.pidAgent_eee import PID_Agent, PidParameters
    from gym_jsbsim.environment_eee import JsbSimEnv_multi_agent
    from gym_jsbsim.wrappers.episodePlotterWrapper_eee import EpisodePlotterWrapper_multi_agent
    import gym_jsbsim.properties as prp
    from gym_jsbsim.reward_funcs_eee import make_glide_angle_reward_components, make_roll_angle_reward_components


    agent_interaction_freq = 5

    pid_elevator_AT = SingleChannel_FlightAgentTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: -6.5},
                                make_base_reward_components= make_glide_angle_reward_components)
    elevator_pid_params = PidParameters( -5e-2, -6.5e-2, -1e-3)
    pid_elevator_agent = PID_Agent('elevator', elevator_pid_params, pid_elevator_AT.get_action_space(), agent_interaction_freq = agent_interaction_freq)

    pid_aileron_AT = SingleChannel_FlightAgentTask('aileron', prp.aileron_cmd, {prp.roll_deg: -15}, max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_reward_components)
    aileron_pid_params = PidParameters(3.5e-2,    1e-2,   0.0)
    pid_aileron_agent = PID_Agent('aileron', aileron_pid_params, pid_aileron_AT.get_action_space(), agent_interaction_freq = agent_interaction_freq)

    agent_task_list = [pid_elevator_AT, pid_aileron_AT]
    trainers = [pid_elevator_agent, pid_aileron_agent]

    env = JsbSimEnv_multi_agent(agent_task_list, agent_interaction_freq = agent_interaction_freq, episode_time_s=120)
    env = EpisodePlotterWrapper_multi_agent(env)

    
    env.set_initial_conditions({prp.initial_flight_path_deg: -1.5}) #just an example, sane defaults are already set in env.__init()__ constructor
    
    obs_n = env.reset()
    pid_elevator_agent.reset_notifier() #only needed for the PID_Agent as it maintains internal state
    pid_aileron_agent.reset_notifier()  #only needed for the PID_Agent as it maintains internal state

    episode_step = 0

    env.showNextPlot(show = True)

    test_net(trainers, env, add_exploration_noise=False)


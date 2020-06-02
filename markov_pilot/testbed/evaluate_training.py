#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import csv
import os
from shutil import copyfile
import numpy as np
import markov_pilot.environment.properties as prp

best_score_n = []  #we don't like globals, but it really helps here
eval_number = 0


def evaluate_training(agent_container, env, lab_journal = None, store_evaluation_experience = True, add_exploration_noise=False, render_mode = None):
    global best_score_n 
    if len(best_score_n) != agent_container.m:
        best_score_n = np.zeros(agent_container.m)

    tgt_flight_path_deg = -6.5
    tgt_roll_angle_deg  = 15
    tgt_sideslip_deg    = 0
    # target_kias = 92  #not used currently as only gliding descent is regulated
    initial_path_angle_gamma_deg = 0
    initial_roll_angle_phi_deg   = 0
    initial_fwd_speed_KAS        = env.aircraft.get_cruise_speed_fps()*0.9 / 1.6878099110965
    initial_aoa_deg              = 1.0  #arbitrary value but not far from a stable flight attitude

    env.change_setpoints({ prp.roll_deg:  tgt_roll_angle_deg,
                            prp.flight_path_deg: tgt_flight_path_deg,
                            prp.sideslip_deg: tgt_sideslip_deg
                        })
    env.set_initial_conditions( { prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                , prp.initial_aoa_deg: initial_aoa_deg
                                })

    env.change_next_episode_length(4*60)    #change the test-episode length to 4 minutes.

    exploration_noise = add_exploration_noise   #to have a handle on that in the debugger
    obs_n = env.reset()
    env.showNextPlot(True, True)
    terminal = False
    score_m = np.zeros(agent_container.m)
    steps = 0
    while not terminal:
        # get action
        actions_n = agent_container.get_action(obs_n, add_exploration_noise=exploration_noise)
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(actions_n) 
        
        if render_mode == 'flightgear' or render_mode == 'timeline':
            env.render(mode = render_mode)

        terminal = env.is_terminal()   #there may be agent independent terminal conditions like the number of episode steps

        if store_evaluation_experience:
            # also store the evaluation experience into the replay buffer. this is valid experience, so use it
            # collect experience, store to per-agent-replay buffers
            agent_experience_m = agent_container.remember(obs_n, actions_n, rew_n, new_obs_n, done_n)
        else:
            agent_experience_m = agent_container.get_per_agent_experience(obs_n, actions_n, rew_n, new_obs_n, done_n)

        obs_n = new_obs_n

        score_m += [exp.rew for exp in agent_experience_m]
        steps += 1
        if steps == int(0.5 * 60 / env.dt): #after half a minute
            tgt_flight_path_deg = -6
            tgt_roll_angle_deg  = 0
            # target_kias = 72

            env.change_setpoints({ prp.roll_deg:  tgt_roll_angle_deg,
                                   prp.flight_path_deg: tgt_flight_path_deg,
                                   prp.sideslip_deg: tgt_sideslip_deg
                                })

        if steps == int(1 *60 / env.dt):    #after a minute
            tgt_flight_path_deg = -7.5
            tgt_roll_angle_deg  = 0
            # target_kias = 100

            env.change_setpoints({ prp.roll_deg:  tgt_roll_angle_deg,
                                   prp.flight_path_deg: tgt_flight_path_deg,
                                   prp.sideslip_deg: tgt_sideslip_deg
                                })

        if steps == int(1.5 *60 / env.dt): #after one and a half minutes
            tgt_flight_path_deg = -7.5
            tgt_roll_angle_deg  = -20
            # target_kias = 80

            env.change_setpoints({ prp.roll_deg:  tgt_roll_angle_deg,
                                   prp.flight_path_deg: tgt_flight_path_deg,
                                   prp.sideslip_deg: tgt_sideslip_deg
                                })

        if steps == int(2.0 *60 / env.dt): #after two minutes
            tgt_flight_path_deg = -7.5
            tgt_roll_angle_deg  = 15
            # target_kias = 80

            env.change_setpoints({ prp.roll_deg:  tgt_roll_angle_deg,
                                   prp.flight_path_deg: tgt_flight_path_deg,
                                   prp.sideslip_deg: tgt_sideslip_deg
                                })

        if steps == int(2.5 *60 / env.dt): #after two and a half minutes
            tgt_flight_path_deg = -7.5
            tgt_roll_angle_deg  = 0
            # target_kias = 80

            env.change_setpoints({ prp.roll_deg:  tgt_roll_angle_deg,
                                   prp.flight_path_deg: tgt_flight_path_deg,
                                   prp.sideslip_deg: tgt_sideslip_deg
                                })

        if steps == int(3 *60 / env.dt): #after three minutes
            tgt_flight_path_deg = -6
            tgt_roll_angle_deg  = 0
            # target_kias = 80

            env.change_setpoints({ prp.roll_deg:  tgt_roll_angle_deg,
                                   prp.flight_path_deg: tgt_flight_path_deg,
                                   prp.sideslip_deg: tgt_sideslip_deg
                                })

        terminal = any(done_n) or terminal
        #env.render()
    print("\tTest yielded a score of : [", end="")
    print(*["%.2f"%sc for sc in score_m], sep = ", ", end="")
    print("].")

    if lab_journal:
        save_last_run(lab_journal, agent_container, score_m)

def save_last_run(lab_journal, agent_container, score_m):
    save_dir = lab_journal.journal_save_dir
    for i, ag in enumerate(agent_container.agents_m):
        eval_dict = {
            'entry_type': ag.name,
            'reward': '{:.2f}'.format(score_m[i]),
            'steps': ag.train_steps,
        }
        save_path = ag.agent_save_path

        #save the agents' state
        filename = f'{ag.name}_rwd-{score_m[i]:06.2f}_steps-{ag.train_steps}.pickle'
        ag.save_agent_state(filename)
        eval_dict.update({'path': os.path.join(save_path, filename)})
        if best_score_n[i] < score_m[i]:
            print("%s: Best score updated: %.3f -> %.3f" % (ag.name, best_score_n[i], score_m[i]))
            bestname = f'{ag.name}_best.pickle'
            copyfile(os.path.join(save_path, filename), os.path.join(save_path, bestname))
            best_score_n[i] = score_m[i]

        lab_journal.append_evaluation_data(eval_dict)


#local test code
if __name__ == '__main__':

    from markov_pilot.environment.environment import NoFGJsbSimEnv_multi
    from markov_pilot.wrappers.episodePlotterWrapper import EpisodePlotterWrapper_multi
    from markov_pilot.tasks.tasks import SingleChannel_FlightTask
    from markov_pilot.agents.AgentTrainer import PID_AgentTrainer, PidParameters
    from markov_pilot.agents.agent_container import AgentContainer, AgentSpec
    from markov_pilot.helper.lab_journal import LabJournal

    #setup an environment
    agent_interaction_freq = 5

    elevator_AT_for_PID = SingleChannel_FlightTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: 0},
                                integral_limit = 100)
                                #integral_limit: self.Ki * dt * int <= output_limit --> int <= 1/0.2*6.5e-2 = 77

    aileron_AT_for_PID = SingleChannel_FlightTask('aileron', prp.aileron_cmd, {prp.roll_deg: 0}, 
                                max_allowed_error= 60, 
                                integral_limit = 100)
                                #integral_limit: self.Ki * dt * int <= output_limit --> int <= 1/0.2*1e-2 = 500

    agent_task_list = [elevator_AT_for_PID, aileron_AT_for_PID]

    env = NoFGJsbSimEnv_multi(agent_task_list, [], agent_interaction_freq = agent_interaction_freq, episode_time_s = 120)  #TODO: task_list is irrelevant for the env!!! REMOVE
    env = EpisodePlotterWrapper_multi(env, output_props=[prp.sideslip_deg])

    #now setup an agent container with simple PID agents
    agent_classes_dict = {
        'PID': PID_AgentTrainer,
    }

    pid_params = {'aileron':  PidParameters(3.5e-2,    1e-2,   0.0),
                  'elevator': PidParameters( -5e-2, -6.5e-2, -1e-3)}

    params_aileron_pid_agent = {
        'pid_params': pid_params['aileron'], 
        'writer': None,
    }

    params_elevator_pid_agent = {
        'pid_params': pid_params['elevator'], 
        'writer': None,
    }

    agent_spec_aileron_PID = AgentSpec('aileron', 'PID', ['aileron'], params_aileron_pid_agent)
    agent_spec_elevator_PID = AgentSpec('elevator', 'PID', ['elevator'], params_elevator_pid_agent)
    agent_spec = [agent_spec_elevator_PID, agent_spec_aileron_PID]

    task_list_n = env.task_list   #we only need the task list to create the mapping. Anything else form the env is not interesting for the agent container.
    agent_container = AgentContainer.init_from_specs(task_list_n, agent_spec, agent_classes_dict, agent_interaction_freq=agent_interaction_freq)

    env.showNextPlot(show = True)

    #now try it with lab journal
    lab_journal = LabJournal('./test_save', {})

    evaluate_training(agent_container, env, lab_journal = lab_journal, add_exploration_noise=False, store_evaluation_experience = False)

    exit(0)
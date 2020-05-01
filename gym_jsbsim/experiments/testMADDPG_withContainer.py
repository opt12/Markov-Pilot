import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import argparse
import time
import os
import csv
import json
import datetime
import pickle
import numpy as np
import importlib

from typing import Union, List

from gym_jsbsim.environment.environment_eee import NoFGJsbSimEnv_multi_agent, JsbSimEnv_multi_agent
from gym_jsbsim.tasks.tasks_eee import SingleChannel_FlightAgentTask
from gym_jsbsim.agents.AgentTrainer import DDPG_AgentTrainer, PID_AgentTrainer, PidParameters, MADDPG_AgentTrainer
from gym_jsbsim.agents.agent_container_eee import AgentContainer, AgentSpec
from gym_jsbsim.wrappers.episodePlotterWrapper_eee import EpisodePlotterWrapper_multi_agent
from gym_jsbsim.wrappers.varySetpointsWrapper import VarySetpointsWrapper
import gym_jsbsim.environment.properties as prp

from gym_jsbsim.experiments.evaluate_training_eee import evaluate_training
from gym_jsbsim.helper.lab_journal import LabJournal

from reward_funcs_eee import make_glide_angle_reward_components, make_roll_angle_reward_components, make_speed_reward_components, make_sideslip_angle_reward_components

## define the initial setpoints
target_path_angle_gamma_deg = -6.5
target_kias = 92
target_roll_angle_phi_deg   = -15
target_sideslip_angle_beta_deg = 0

def parse_args():   #TODO: adapt this. Taken from https://github.com/openai/maddpg/
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    # parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len-sec", type=int, default=120, help="maximum episode length in seconds (steps = seconds*interaction frequ.)")
    parser.add_argument("--num-steps", type=int, default=30000, help="number of training steps to perfoem")
    # parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    # parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    # parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--interaction-frequency", type=float, default=5, help="frequency of agent interactions with the environment")
    # Core training parameters
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="learning rate for the actor training Adam optimizer")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate for the critic training Adam optimizer")
    parser.add_argument("--tau", type=float, default=1e-3, help="target network adaptation factor")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="number of episodes to optimize at the same time")
    parser.add_argument("--replay-size", type=int, default=1000000, help="size of the replay buffer")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='Default_Experiment', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", nargs='+', type=int, default=False)    #to restore agents and env from lab-journal lines given as list and continue training
    parser.add_argument("--play", nargs='+', type=int, default=False)    #to play with agents and env restored from lab-journal lines
    parser.add_argument("--best", type=bool, default=False)    #when given, the first line form restore or play will be used to restore the environment and the best agents for that run will be loaded
    # TODO: --flightgear
    parser.add_argument("--flightgear", type=bool, default=False)    #when given, together with --play [lines] the environment will be replaced with the flight-gear enabled and the player will render to FlightGear
    # parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--testing-iters", type=int, default=2000, help="number of steps before running a performance test")
    #parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--base-dir", type=str, default="./", help="directory the test_run date is saved")
    return parser.parse_args()

def setup_env(arglist) -> NoFGJsbSimEnv_multi_agent:
    agent_interaction_freq = arglist.interaction_frequency
    episode_time_s=arglist.max_episode_len_sec

    ## define the initial conditions
    initial_path_angle_gamma_deg = target_path_angle_gamma_deg + 3
    initial_roll_angle_phi_deg   = target_roll_angle_phi_deg + 10
    initial_sideslip_angle_beta_deg   = 0
    initial_fwd_speed_KAS        = 80
    initial_aoa_deg              = 1.0
    # initial_altitude_ft          = 6000

    elevator_AT_for_PID = SingleChannel_FlightAgentTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: target_path_angle_gamma_deg},
                                integral_limit = 100)
                                #integral_limit: self.Ki * dt * int <= output_limit --> int <= 1/0.2*6.5e-2 = 77

    aileron_AT_for_PID = SingleChannel_FlightAgentTask('aileron', prp.aileron_cmd, {prp.roll_deg: initial_roll_angle_phi_deg}, 
                                max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_reward_components,
                                integral_limit = 100)
                                #integral_limit: self.Ki * dt * int <= output_limit --> int <= 1/0.2*1e-2 = 500

    rudder_AT_for_PID = SingleChannel_FlightAgentTask('rudder', prp.rudder_cmd, {prp.sideslip_deg: 0},
                                max_allowed_error= 10, 
                                make_base_reward_components= make_sideslip_angle_reward_components,
                                integral_limit = 100)
                                #integral_limit: self.Ki * dt * int <= output_limit --> int <= 1/0.2*1e-2 = 500

    elevator_AT = SingleChannel_FlightAgentTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: target_path_angle_gamma_deg},
                                presented_state=[prp.elevator_cmd, prp.q_radps, prp.indicated_airspeed],
                                max_allowed_error= 30, 
                                make_base_reward_components= make_glide_angle_reward_components,
                                integral_limit = 0.5)

    elevator_Speed_AT = SingleChannel_FlightAgentTask('elevator', prp.elevator_cmd, {prp.indicated_airspeed: target_kias},
                                presented_state=[prp.elevator_cmd, prp.q_radps],
                                max_allowed_error= 50, 
                                make_base_reward_components= make_speed_reward_components,
                                integral_limit = 2)

    elevator_Speed_AT_for_PID = SingleChannel_FlightAgentTask('elevator', prp.elevator_cmd, {prp.indicated_airspeed: target_kias},
                                presented_state=[prp.elevator_cmd, prp.q_radps],
                                max_allowed_error= 50, 
                                make_base_reward_components= make_speed_reward_components,
                                integral_limit = 200)


    aileron_AT = SingleChannel_FlightAgentTask('aileron', prp.aileron_cmd, {prp.roll_deg: initial_roll_angle_phi_deg}, 
                                presented_state=[prp.aileron_cmd, prp.p_radps, prp.indicated_airspeed],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_reward_components,
                                integral_limit = 0.25)

    rudder_AT = SingleChannel_FlightAgentTask('rudder', prp.rudder_cmd, {prp.sideslip_deg: 0}, 
                                presented_state=[prp.rudder_cmd, prp.aileron_cmd, prp.r_radps, prp.p_radps, prp.indicated_airspeed],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_sideslip_angle_reward_components,
                                integral_limit = 0.25)

    agent_task_list = [elevator_AT, aileron_AT]
    # agent_task_list = [elevator_AT, aileron_AT, rudder_AT]
    # agent_task_types = ['PID', 'PID']
    # agent_task_types = ['PID', 'DDPG']  #TODO: This is irrelevant for the env!!! REMOVE
    # agent_task_types = ['DDPG', 'MADDPG']
    # agent_task_types = ['MADDPG', 'MADDPG']

    env = NoFGJsbSimEnv_multi_agent(agent_task_list, [], agent_interaction_freq = agent_interaction_freq, episode_time_s = episode_time_s)  #TODO: task_list is irrelevant for the env!!! REMOVE
    env = EpisodePlotterWrapper_multi_agent(env, output_props=[prp.sideslip_deg])

    env.set_initial_conditions({ prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                    , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                    , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                    , prp.initial_aoa_deg: initial_aoa_deg
                                    # , prp.initial_altitude_ft: initial_altitude_ft
                                    }) #just an example, sane defaults are already set in env.__init()__ constructor

    env.set_meta_information(experiment_name = arglist.exp_name)
    return env

def setup_container_from_env(env, arglist):
    
    agent_classes_dict = {
        'PID': PID_AgentTrainer,
        'MADDPG': MADDPG_AgentTrainer,
        'DDPG': DDPG_AgentTrainer,
    }

    #for PID controllers we need an elaborated parameter set for each type
    pid_params = {'aileron':  PidParameters(3.5e-2,    1e-2,   0.0),
                  'elevator': PidParameters( -5e-2, -6.5e-2, -1e-3),
                  'rudder':   PidParameters(  0.6e-1, 0, 0),            #TODO: This parameter set does'nt work really good for coordinated turns
                  'elevator_speed': PidParameters( 2e-3, 6.5e-3, 1e-4), #TODO: This parameter set does'nt work at all for speed control
                  }    

    params_aileron_pid_agent = {
        'pid_params': pid_params['aileron'], 
        'writer': None,
    }

    params_elevator_pid_agent = {
        'pid_params': pid_params['elevator'], 
        'writer': None,
    }

    params_rudder_pid_agent = {
        'pid_params': pid_params['rudder'], 
        'writer': None,
    }
    
    params_elevator_speed_pid_agent = {
        'pid_params': pid_params['elevator_speed'], 
        'writer': None,
    }

    #for the learning agents, a standard parameter set will do; the details will be learned
    params_DDPG_MADDPG_agent = {
        **vars(arglist),
        # 'layer1_size': 800,
        # 'layer2_size': 600,
        'writer': None,
    }

    agent_spec_aileron_PID = AgentSpec('aileron', 'PID', ['aileron'], params_aileron_pid_agent)

    agent_spec_aileron_DDPG = AgentSpec('aileron', 'DDPG', ['aileron'], params_DDPG_MADDPG_agent)

    agent_spec_aileron_MADDPG = AgentSpec('aileron', 'MADDPG', ['aileron'], params_DDPG_MADDPG_agent)

    agent_spec_elevator_PID = AgentSpec('elevator', 'PID', ['elevator'], params_elevator_pid_agent)
    agent_spec_elevator_speed_PID = AgentSpec('elevator', 'PID', ['elevator'], params_elevator_speed_pid_agent)

    agent_spec_elevator_DDPG = AgentSpec('elevator', 'DDPG', ['elevator'], params_DDPG_MADDPG_agent)

    agent_spec_elevator_MADDPG = AgentSpec('elevator', 'MADDPG', ['elevator'], params_DDPG_MADDPG_agent)

    agent_spec_rudder_MADDPG = AgentSpec('rudder', 'MADDPG', ['rudder'], params_DDPG_MADDPG_agent)
    agent_spec_rudder_DDPG = AgentSpec('rudder', 'DDPG', ['rudder'], params_DDPG_MADDPG_agent)
    agent_spec_rudder_PID = AgentSpec('rudder', 'PID', ['rudder'], params_rudder_pid_agent)

    agent_spec_elevator_aileron_DDPG = AgentSpec('elevator_aileron', 'DDPG', ['elevator', 'aileron'], params_DDPG_MADDPG_agent)

    #Here we specify which agents shall be initiated; chose form the above defined single-specs
    # agent_spec = [agent_spec_elevator_MADDPG, agent_spec_aileron_MADDPG, agent_spec_rudder_MADDPG]
    # agent_spec = [agent_spec_elevator_aileron_DDPG]
    agent_spec = [agent_spec_elevator_MADDPG, agent_spec_aileron_MADDPG]
    # agent_spec = [agent_spec_elevator_PID, agent_spec_aileron_PID, agent_spec_rudder_DDPG]

    task_list_n = env.task_list   #we only need the task list to create the mapping. Anything else form the env is not interesting for the agent container.
    agent_container = AgentContainer.init_from_env(task_list_n, agent_spec, agent_classes_dict, **vars(arglist))

    return agent_container

def save_test_run(env: JsbSimEnv_multi_agent, agent_container: AgentContainer, lab_journal: LabJournal, arglist):
    """
    - creates a suitable directory for the test run
    - adds a sidecar file containing the meta information on the run (dict saved as pickle)
    - adds a text file containing the meta information on the run
    - add a line to the global csv-file for the test run
    """
    # IMPORTANT to do this first,  to make the lab_journal aware of the start time of the run
    lab_journal.set_run_start()

    task_names = '_'.join([t.name for t in env.task_list])
    date = lab_journal.run_start.strftime("%Y_%m_%d")
    time = lab_journal.run_start.strftime("%H-%M")

    #build the path name for the run protocol
    save_path = os.path.join(lab_journal.journal_save_dir, env.aircraft.name, arglist.exp_name, task_names, date+'-'+time)

    #create the base directory for this test_run
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #create the directories for each agent_task
    for a in agent_container.agents_m:
        agent_path = os.path.join(save_path, a.name)
        os.makedirs(os.path.join(save_path, a.name), exist_ok=True)
        a.set_save_path(agent_path)

    env.save_env_data(arglist, save_path)
    agent_container.save_agent_container_data(save_path)
    #eventually append the run data to the csv-file. 
    csv_line_nr = lab_journal.append_run_data(env, agent_container.agents_m, save_path)
    env.set_meta_information(csv_line_nr = csv_line_nr)

def restore_env_from_journal(lab_journal, line_numbers: Union[int, List[int]]) -> NoFGJsbSimEnv_multi_agent:
    ENV_PICKLE = 'environment_init.pickle'  #these are hard default names for the files
    TASKS_PICKLE ='task_agent.pickle'       #these are hard default names for the files

    ln = line_numbers if isinstance(line_numbers, int) else line_numbers[0]

    #get run protocol
    try:
        model_file = lab_journal.get_model_filename(ln)
        run_protocol_path = lab_journal.find_associated_run_path(model_file)
    except TypeError:
        print(f"there was no run protocol found that is associated with line_number {ln}")
        exit()

    if run_protocol_path == None:
        print(f"there was no run protocol found that is associated with line_number {ln}")
        exit()

    #load the TASKS_PICKLE and restore the task_list
    with open(os.path.join(run_protocol_path, TASKS_PICKLE), 'rb') as infile:
        task_agent_data = pickle.load(infile)
    
    task_agents = []
    for idx in range(len(task_agent_data['task_list_class_names'])):
        task_list_init = task_agent_data['task_list_init'][idx]
        task_list_class_name = task_agent_data['task_list_class_names'][idx]
        make_base_reward_components_file = task_agent_data['make_base_reward_components_file'][idx]
        make_base_reward_components_fn = task_agent_data['make_base_reward_components_fn'][idx]

        #load the make_base_reward_components function from a python-file
        #load function from given filepath  https://stackoverflow.com/a/67692/2682209
        spec = importlib.util.spec_from_file_location("make_base_rwd", os.path.join(run_protocol_path, make_base_reward_components_file))
        func_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(func_module)
        make_base_reward_components = getattr(func_module, make_base_reward_components_fn)

        #get the class for the task_agent https://stackoverflow.com/a/17960039/2682209
        class_ = getattr(sys.modules[__name__], task_list_class_name)
        #add the make_base_reward_components function to the parameter dict
        task_list_init.update({'make_base_reward_components': make_base_reward_components})
        #transform the setpoint_props and setpoint_values lists to setpoints dict
        task_list_init.update({'setpoints': dict(zip(task_list_init['setpoint_props'], task_list_init['setpoint_values']))})
        del task_list_init['setpoint_props']
        del task_list_init['setpoint_values']
        ta = class_(**task_list_init)
        task_agents.append(ta)

    #the task_agents are now ready, so now let's prepare the environment
    #load the ENV_PICKLE and restore the task_list
    with open(os.path.join(run_protocol_path, ENV_PICKLE), 'rb') as infile:
        env_data = pickle.load(infile)

    env_init_dicts = env_data['init_dicts']
    env_classes = env_data['env_classes']

    #create the innermost environment with the task_list added
    #load the env class
    env_class_ = getattr(sys.modules[__name__], env_classes[0])
    env_init = env_init_dicts[0]
    env_init.update({'task_list':task_agents})

    env = env_class_(**env_init)
    
    #apply wrappers if available
    #TODO: what about other wrappers than EpisodePlotterWrapper? We don't save the VarySetpointWrapper to the wrappers list.
    for idx in range(1, len(env_init_dicts)):
        wrapper_class_ = getattr(sys.modules[__name__], env_classes[idx])
        wrap_init = env_init_dicts[idx]
        wrap_init.update({'env': env})
        env = wrapper_class_(**wrap_init)
    
    env.set_meta_information(csv_line_nr = ln)  #set the line number, the environment was loaded from
    return env

def restore_agent_container_from_journal(lab_journal, line_numbers: Union[int, List[int]]) -> 'AgentContainer':
    CONTAINER_PICKLE = 'agent_container.pickle'

    ln = [line_numbers] if isinstance(line_numbers, int) else line_numbers

    #get run protocol

    agent_pickle_files_m = [lab_journal.get_model_filename(line) for line in ln]
    try:
        model_file = lab_journal.get_model_filename(ln[0])
        run_protocol_path = lab_journal.find_associated_run_path(model_file)
    except TypeError:
        print(f"there was no run protocol found that is associated with line_number {ln}")
        exit()

    agent_container = AgentContainer.init_from_save(os.path.join(run_protocol_path, CONTAINER_PICKLE), agent_pickle_files_m)

    return agent_container

def perform_training(training_env: JsbSimEnv_multi_agent, testing_env: JsbSimEnv_multi_agent, agent_container: AgentContainer, arglist: argparse.Namespace):

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(len(agent_container.agents_m))]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    # saver = tf.train.Saver()  #TODO: need to add some save/restore code compatible to pytorch
    obs_n = training_env.reset()
    episode_step = 0
    episode_counter = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # get action
        actions_n = agent_container.get_action(obs_n, add_exploration_noise=True)
        # environment step
        new_obs_n, rew_n, done_n, _ = training_env.step(actions_n)   #no need to process info_n
        episode_step += 1
        done = any(done_n)  #we end the episode if any of the involved tasks came to an end
        terminal = training_env.is_terminal()   #there may be agent independent terminal conditions like the number of episode steps
        # collect experience, store to per-agent-replay buffers
        agent_experience_m = agent_container.remember(obs_n, actions_n, rew_n, new_obs_n, done_n)
        obs_n = new_obs_n

        #track episode and agent rewards
        for i, exp in enumerate(agent_experience_m):         #there should be some np-magic to convert to a one liner
            episode_rewards[-1] += exp.rew      #overall reward as sum of all agent rewards
            agent_rewards[i][-1] += exp.rew

        # perform an actual training step
        agent_container.train_agents()

        # increment global step counter
        train_step += 1

        #do some housekeeping when episode is over
        if done or terminal:        #episode is over
            episode_counter += 1
            obs_n = training_env.reset()    #start new episode
            
            showPlot = False    #this is here for debugging purposes, you can enable plotting for one time
            training_env.showNextPlot(show = showPlot)

            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)

        # progress indicator
        if (train_step) % (arglist.testing_iters/20) == 0:
            print('.', end='', flush=True)
        # every arglist.testing_iters training steps, an evaluation run is started in the testing_env
        if (train_step) % arglist.testing_iters == 0:
            print('')
            t_end = time.time()
            print(f"train_step {train_step}, Episode {episode_counter}: performed {arglist.testing_iters} steps in {t_end-t_start:.2f} seconds; that's {arglist.testing_iters/(t_end-t_start):.2f} steps/sec")
            testing_env.set_meta_information(episode_number = episode_counter)
            testing_env.set_meta_information(train_step = train_step)
            testing_env.showNextPlot(True)
            evaluate_training(agent_container, testing_env, lab_journal=lab_journal, add_exploration_noise=False)    #run the standardized test on the test_env TODO: add lab_journal again
            t_start = time.time()

        # env.render(mode='flightgear') #not really useful in training

        #TODO: check the useful outputs for tracking the training progress
        # save model, display training output   
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):    #save every arglist.save_rate completed episodes    
            # Keep track of final episode reward    TODO: check which outputs are really needed
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if train_step > arglist.num_steps:
            rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break

if __name__ == '__main__':

    arglist = parse_args()

    lab_journal = LabJournal(arglist.base_dir, arglist)

    # testing_env = restore_env_from_journal(lab_journal, 91)

    # testing_env = VarySetpointsWrapper(testing_env, prp.roll_deg, (-30, 30), (10, 120), (5, 30))#, (0.05, 0.5))
    # testing_env = VarySetpointsWrapper(testing_env, prp.flight_path_deg, (-10, -5.5), (10, 120), (5, 30))#, (0.05, 0.5))

    # agent_container = restore_agent_container_from_journal(lab_journal, [91, 90,101])
    # evaluate_training(agent_container, testing_env, lab_journal=None, add_exploration_noise=False)    #run the standardized test on the test_env
    # exit(0)


    # exit(0)

    training_env = setup_env(arglist)
    testing_env = setup_env(arglist)

    #apply Varyetpoints to the training to increase the variance of training data
    training_env = VarySetpointsWrapper(training_env, prp.roll_deg, (-30, 30), (10, 30), (5, 30), (0.05, 0.5))
    training_env = VarySetpointsWrapper(training_env, prp.flight_path_deg, (-10, -5.5), (10, 45), (5, 30), (0.05, 0.5))
    # training_env = VarySetpointsWrapper(training_env, prp.sideslip_deg, (-3, 3), (10, 45), (5, 30), (0.05, 0.5))


    agent_container = setup_container_from_env(training_env, arglist)

    save_test_run(testing_env, agent_container, lab_journal, arglist)  #use the testing_env here to have the save_path available in the evaluation

    perform_training(training_env, testing_env, agent_container, arglist)
    
    training_env.close()
    testing_env.close()


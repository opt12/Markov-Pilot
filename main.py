import sys, os            
# sys.path.append(os.path.join(os.path.dirname(__file__)) #TODO: Is this a good idea? Dunno! It works!

# print(os.path.join(os.path.dirname(__file__)))

import argparse

import markov_pilot.environment.properties as prp

from markov_pilot.environment.environment import NoFGJsbSimEnv_multi, JsbSimEnv_multi
from markov_pilot.wrappers.episodePlotterWrapper import EpisodePlotterWrapper_multi
from markov_pilot.wrappers.varySetpointsWrapper import VarySetpointsWrapper

from markov_pilot.tasks.tasks import SingleChannel_FlightTask, SingleChannel_MinimumProps_Task

from reward_funcs import _make_base_reward_components, \
                            make_glide_angle_reward_components, make_roll_angle_reward_components, make_speed_reward_components, make_sideslip_angle_reward_components, \
                            make_glide_path_angle_reward_components, make_elevator_actuation_reward_components, \
                            make_roll_angle_error_only_reward_components, make_roll_angle_error_punish_actuation_reward_components, make_roll_angle_integral_reward_components, make_roll_angle_integral_reward_components, \
                            make_angular_error_only_reward_components, make_angular_error_punish_actuation_reward_components, make_angular_integral_reward_components, make_angular_derivative_integral_reward_components, \
                            make_rudder_reward_components

from markov_pilot.agents.AgentTrainer import DDPG_AgentTrainer, PID_AgentTrainer, PidParameters, MADDPG_AgentTrainer
from markov_pilot.agents.agent_container import AgentContainer, AgentSpec
from markov_pilot.agents.train import perform_training

from markov_pilot.helper.lab_journal import LabJournal
from markov_pilot.helper.load_store import restore_agent_container_from_journal, restore_env_from_journal, save_test_run

from markov_pilot.testbed.evaluate_training import evaluate_training

## define the initial setpoints
target_path_angle_gamma_deg = -6.5
target_kias = 92
target_roll_angle_phi_deg   = -15
target_sideslip_angle_beta_deg = 0

def parse_args():   #used https://github.com/openai/maddpg/ as a basis
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--max-episode-len-sec", type=int, default=120, help="maximum episode length in seconds (steps = seconds*interaction frequ.)")
    parser.add_argument("--num-steps", type=int, default=30000, help="number of training steps to perform")
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
    parser.add_argument("--best", type=bool, default=False)    #TODO: when given, the first line from restore or play will be used to restore the environment and the best agents for that run will be loaded 
    parser.add_argument("--flightgear", type=bool, default=False)    #TODO: when given, together with --play [lines] the environment will be replaced with the flight-gear enabled and the player will render to FlightGear
    parser.add_argument("--testing-iters", type=int, default=2000, help="number of steps before running a performance test")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--base-dir", type=str, default="./", help="directory the test_run date is saved")
    return parser.parse_args()

def setup_env(arglist) -> NoFGJsbSimEnv_multi:
    agent_interaction_freq = arglist.interaction_frequency
    episode_time_s=arglist.max_episode_len_sec

    ## define the initial conditions
    initial_path_angle_gamma_deg = target_path_angle_gamma_deg + 3
    initial_roll_angle_phi_deg   = target_roll_angle_phi_deg + 10
    initial_sideslip_angle_beta_deg   = 0
    initial_fwd_speed_KAS        = 80
    initial_aoa_deg              = 1.0
    # initial_altitude_ft          = 6000

    elevator_AT_for_PID = SingleChannel_FlightTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: target_path_angle_gamma_deg},
                                make_base_reward_components=make_angular_integral_reward_components,   #pass this in here as otherwise, the restore form disk gets nifty
                                integral_limit = 100)
                                #integral_limit: self.Ki * dt * int <= output_limit --> int <= 1/0.2*6.5e-2 = 77

    aileron_AT_for_PID = SingleChannel_FlightTask('aileron', prp.aileron_cmd, {prp.roll_deg: initial_roll_angle_phi_deg}, 
                                make_base_reward_components=make_angular_integral_reward_components,   #pass this in here as otherwise, the restore form disk gets nifty
                                integral_limit = 100)
                                #integral_limit: self.Ki * dt * int <= output_limit --> int <= 1/0.2*1e-2 = 500

    rudder_AT_for_PID = SingleChannel_FlightTask('rudder', prp.rudder_cmd, {prp.sideslip_deg: 0},
                                max_allowed_error= 10, 
                                make_base_reward_components=_make_base_reward_components,   #pass this in here as otherwise, the restore form disk gets nifty
                                integral_limit = 100)
                                #integral_limit: self.Ki * dt * int <= output_limit --> int <= 1/0.2*1e-2 = 500

    elevator_AT = SingleChannel_FlightTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: target_path_angle_gamma_deg},
                                presented_state=[prp.elevator_cmd, prp.q_radps, prp.indicated_airspeed],
                                max_allowed_error= 30, 
                                make_base_reward_components= make_glide_angle_reward_components,
                                integral_limit = 0.5)

    glide_path_task = SingleChannel_FlightTask('glide_path_task', [], {prp.flight_path_deg: target_path_angle_gamma_deg},
                                presented_state=[prp.elevator_cmd, prp.q_radps, prp.indicated_airspeed],
                                max_allowed_error= 30, 
                                make_base_reward_components= make_glide_path_angle_reward_components,
                                integral_limit = 0.5)

    elevator_actuation_task = SingleChannel_FlightTask('elevator_actuation_task', prp.elevator_cmd, {},
                                presented_state=[prp.elevator_cmd],
                                make_base_reward_components= make_elevator_actuation_reward_components)

    elevator_Speed_AT = SingleChannel_FlightTask('elevator', prp.elevator_cmd, {prp.indicated_airspeed: target_kias},
                                presented_state=[prp.elevator_cmd, prp.q_radps],
                                max_allowed_error= 50, 
                                make_base_reward_components= make_speed_reward_components,
                                integral_limit = 2)

    elevator_Speed_AT_for_PID = SingleChannel_FlightTask('elevator', prp.elevator_cmd, {prp.indicated_airspeed: target_kias},
                                presented_state=[prp.elevator_cmd, prp.q_radps],
                                max_allowed_error= 50, 
                                make_base_reward_components= make_speed_reward_components,
                                integral_limit = 200)


    aileron_AT = SingleChannel_FlightTask('aileron', prp.aileron_cmd, {prp.roll_deg: initial_roll_angle_phi_deg}, 
                                presented_state=[prp.aileron_cmd, prp.p_radps, prp.indicated_airspeed],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_reward_components,
                                integral_limit = 0.25)

    rudder_AT = SingleChannel_FlightTask('rudder', prp.rudder_cmd, {prp.sideslip_deg: 0}, 
                                presented_state=[prp.rudder_cmd, prp.r_radps, prp.p_radps, prp.indicated_airspeed, 
                                                # aileron_AT.prop_error   #TODO: this relies on defining aileron_AT before rudder_AT :-()
                                                ],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_sideslip_angle_reward_components,
                                integral_limit = 0.25)

    velocity_AT = SingleChannel_FlightTask('velocity_kias', prp.throttle_cmd, {prp.indicated_airspeed: target_kias}, 
                                presented_state=[prp.throttle_cmd],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_speed_reward_components,
                                integral_limit = 0.25)


    agent_task_list = [elevator_AT, aileron_AT, rudder_AT]

    aileron_Exp1_1 = SingleChannel_FlightTask('aileron', prp.aileron_cmd, {prp.roll_deg: initial_roll_angle_phi_deg}, 
                                presented_state=[],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_error_only_reward_components,
                                integral_limit = 0.25)

    aileron_Exp1_2 = SingleChannel_FlightTask('aileron', prp.aileron_cmd, {prp.roll_deg: initial_roll_angle_phi_deg}, 
                                presented_state=[prp.p_radps, prp.indicated_airspeed],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_error_only_reward_components,
                                integral_limit = 0.25)

    aileron_Exp1_3 = SingleChannel_FlightTask('aileron', prp.aileron_cmd, {prp.roll_deg: initial_roll_angle_phi_deg}, 
                                presented_state=[prp.p_radps, prp.aileron_cmd],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_error_punish_actuation_reward_components,
                                integral_limit = 0.25)

    aileron_Exp1_6 = SingleChannel_FlightTask('aileron', prp.aileron_cmd, {prp.roll_deg: initial_roll_angle_phi_deg}, 
                                presented_state=[prp.p_radps, prp.aileron_cmd, prp.indicated_airspeed],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_error_punish_actuation_reward_components,
                                integral_limit = 0.25)

    aileron_Exp2_0 = SingleChannel_FlightTask('aileron', prp.aileron_cmd, {prp.roll_deg: initial_roll_angle_phi_deg}, 
                                presented_state=[prp.p_radps, prp.aileron_cmd, prp.indicated_airspeed],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_integral_reward_components,
                                integral_limit = 0.25)

    elevator_Exp1_0 = SingleChannel_FlightTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: initial_path_angle_gamma_deg}, 
                                presented_state=[prp.q_radps, prp.indicated_airspeed], 
                                max_allowed_error= 30, 
                                make_base_reward_components= make_angular_error_only_reward_components,
                                integral_limit = 0.25)

    elevator_Exp2_0 = SingleChannel_FlightTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: initial_path_angle_gamma_deg}, 
                                presented_state=[prp.q_radps, prp.indicated_airspeed, prp.elevator_cmd],
                                max_allowed_error= 30, 
                                make_base_reward_components= make_angular_error_punish_actuation_reward_components,
                                integral_limit = 0.25)

    elevator_Exp3_0 = SingleChannel_FlightTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: initial_path_angle_gamma_deg}, 
                                presented_state=[prp.q_radps, prp.indicated_airspeed, prp.elevator_cmd],
                                max_allowed_error= 30, 
                                make_base_reward_components= make_angular_integral_reward_components,
                                integral_limit = 0.25)

    elevator_Exp4_0 = SingleChannel_FlightTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: target_path_angle_gamma_deg}, 
                                presented_state=[prp.q_radps, prp.indicated_airspeed, prp.elevator_cmd],# prp.rudder_cmd, prp.aileron_cmd],
                                max_allowed_error= 30, 
                                make_base_reward_components= make_angular_integral_reward_components,
                                integral_limit = 0.25)

    aileron_Exp5_0 = SingleChannel_FlightTask('aileron', prp.aileron_cmd, {prp.roll_deg: target_roll_angle_phi_deg}, 
                                presented_state=[prp.p_radps, prp.indicated_airspeed, prp.aileron_cmd],# prp.elevator_cmd, prp.aileron_cmd],
                                max_allowed_error= 60, 
                                make_base_reward_components= make_angular_integral_reward_components,
                                integral_limit = 0.25)

    rudder_Exp9_0 = SingleChannel_FlightTask('rudder', prp.rudder_cmd, {prp.sideslip_deg: target_sideslip_angle_beta_deg}, 
                                presented_state=[prp.r_radps, prp.indicated_airspeed, prp.rudder_cmd,# prp.aileron_cmd, prp.elevator_cmd,
                                aileron_Exp5_0.setpoint_value_props[0], aileron_Exp5_0.setpoint_props[0]],   #TODO: this relies on defining aileron_Exp5_0 before rudder_AT :-()
                                max_allowed_error= 30, 
                                # make_base_reward_components= make_sideslip_angle_reward_components,
                                make_base_reward_components= make_sideslip_angle_reward_components,
                                integral_limit = 0.25)


    agent_task_list = [elevator_Exp4_0, aileron_Exp5_0, rudder_Exp9_0]
    # agent_task_list = [elevator_AT_for_PID, aileron_AT_for_PID, rudder_Exp9_0]
    # agent_task_list = [elevator_Exp4_0, aileron_Exp5_0, rudder_AT_for_PID]

    # agent_task_list = [elevator_actuation_task, glide_path_task, aileron_AT]
    # agent_task_list = [elevator_AT_for_PID, aileron_AT]
    
    # agent_task_list = [elevator_AT_for_PID, aileron_AT_full_state_dev_only]

    # agent_task_list = [elevator_AT, aileron_AT, rudder_AT]

    env = NoFGJsbSimEnv_multi(agent_task_list, agent_interaction_freq = agent_interaction_freq, episode_time_s = episode_time_s)
    env = EpisodePlotterWrapper_multi(env, output_props=[prp.sideslip_deg])

    env.set_initial_conditions({ prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                    , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                    , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                    , prp.initial_aoa_deg: initial_aoa_deg
                                    # , prp.initial_altitude_ft: initial_altitude_ft
                                    }) #just an example, sane defaults are already set in env.__init()__ constructor

    env.set_meta_information(experiment_name = arglist.exp_name)
    return env

def setup_container(task_list, arglist):
    
    agent_classes_dict = {
        'PID': PID_AgentTrainer,
        'MADDPG': MADDPG_AgentTrainer,
        'DDPG': DDPG_AgentTrainer,
    }

    #for PID controllers we need an elaborated parameter set for each type
    pid_params = {'aileron':  PidParameters(3.5e-2,    1e-2,   0.0),
                  'elevator': PidParameters( -5e-2, -6.5e-2, -1e-3),
                  'rudder':   PidParameters(  0, 0, 0),            #TODO: This parameter set just leaves the rudder alone. No actuation at all
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
        'layer1_size': 400,
        'layer2_size': 300,

        'writer': None,
    }
    #for the learning agents, a standard parameter set will do; the details will be learned
    params_DDPG_MADDPG_agent_big_net = {
        **vars(arglist),
        'layer1_size': 1200,
        'layer2_size': 900,

        'writer': None,
    }

    agent_spec_aileron_PID = AgentSpec('aileron', 'PID', ['aileron'], params_aileron_pid_agent)

    agent_spec_aileron_DDPG = AgentSpec('aileron', 'DDPG', ['aileron'], params_DDPG_MADDPG_agent)

    agent_spec_aileron_MADDPG = AgentSpec('aileron', 'MADDPG', ['aileron'], params_DDPG_MADDPG_agent)

    agent_spec_elevator_PID = AgentSpec('elevator', 'PID', ['elevator'], params_elevator_pid_agent)
    agent_spec_elevator_speed_PID = AgentSpec('elevator', 'PID', ['elevator'], params_elevator_speed_pid_agent)

    agent_spec_elevator_DDPG = AgentSpec('elevator', 'DDPG', ['elevator'], params_DDPG_MADDPG_agent)

    agent_spec_elevator_MADDPG = AgentSpec('elevator', 'MADDPG', ['elevator'], params_DDPG_MADDPG_agent)

    params_DDPG_MADDPG_separated_agent = {
        **vars(arglist),
        # 'layer1_size': 800,
        # 'layer2_size': 600,
        'task_reward_weights': [2, 14],
        'writer': None,
    }
    
    agent_spec_elevator_MADDPG_separated_tasks = AgentSpec('elevator', 'MADDPG', ['elevator_action', 'glidepath_setpoint'], params_DDPG_MADDPG_separated_agent)
    agent_spec_elevator_DDPG_velocity_glide_path = AgentSpec('thrust_elevator', 'DDPG', ['velocity_kias', 'elevator'], params_DDPG_MADDPG_separated_agent)

    agent_spec_rudder_MADDPG = AgentSpec('rudder', 'MADDPG', ['rudder'], params_DDPG_MADDPG_agent_big_net)
    agent_spec_rudder_DDPG = AgentSpec('rudder', 'DDPG', ['rudder'], params_DDPG_MADDPG_agent)
    agent_spec_rudder_PID = AgentSpec('rudder', 'PID', ['rudder'], params_rudder_pid_agent)

    agent_spec_glide_path_MADDPG_separated_tasks = AgentSpec('elevator', 'MADDPG', ['elevator_actuation_task', 'glide_path_task'], params_DDPG_MADDPG_separated_agent)
    agent_spec_elevator_aileron_DDPG = AgentSpec('elevator_aileron', 'DDPG', ['elevator', 'aileron'], params_DDPG_MADDPG_agent)

    agent_spec_aileron_rudder_MADDPG = AgentSpec('aileron_rudder', 'MADDPG', ['aileron', 'rudder'], params_DDPG_MADDPG_agent_big_net)
    agent_spec_elevator_aileron_rudder_MADDPG = AgentSpec('ele_ail_rud', 'DDPG', ['elevator', 'aileron', 'rudder'], params_DDPG_MADDPG_agent_big_net)


    #Here we specify which agents shall be initiated; chose form the above defined single-specs
    # agent_spec = [agent_spec_elevator_MADDPG, agent_spec_aileron_MADDPG, agent_spec_rudder_MADDPG]
    # agent_spec = [agent_spec_elevator_aileron_DDPG]
    agent_spec = [agent_spec_elevator_DDPG, agent_spec_aileron_DDPG, agent_spec_rudder_DDPG]
    agent_spec = [agent_spec_elevator_PID, agent_spec_aileron_PID, agent_spec_rudder_DDPG]

    # agent_spec = [agent_spec_elevator_aileron_DDPG, agent_spec_rudder_PID]
    # agent_spec = [agent_spec_elevator_PID, agent_spec_aileron_MADDPG]

    # agent_spec = [agent_spec_elevator_MADDPG, agent_spec_aileron_MADDPG, agent_spec_rudder_MADDPG]

    task_list_n = task_list   #we only need the task list to create the mapping. Anything else form the env is not interesting for the agent container.
    agent_container = AgentContainer.init_from_specs(task_list_n, agent_spec, agent_classes_dict, **vars(arglist))

    return agent_container

if __name__ == '__main__':

    arglist = parse_args()

    lab_journal = LabJournal(arglist.base_dir, arglist)

    # pid_container = setup_PID_benchmark_container(pid_env_benchmark.task_list, arglist)

    restore_lines = [4674, 4721, 4768]
    # testing_env = restore_env_from_journal(lab_journal, restore_lines[0], target_environment='FG')
    # testing_env = restore_env_from_journal(lab_journal, restore_lines[0])

    testing_env = setup_env(arglist)


    # testing_env = VarySetpointsWrapper(testing_env, prp.roll_deg, (-30, 30), (10, 120), (5, 30), (0.05, 0.1))
    # testing_env = VarySetpointsWrapper(testing_env, prp.flight_path_deg, (-9, -5.5), (10, 120), (5, 30), (0.05, 0.1))

    agent_container = restore_agent_container_from_journal(lab_journal, restore_lines)#, task_list_n=testing_env.task_list, mapping_dict= mapping_dict)

    # save_test_run(testing_env, agent_container, lab_journal, arglist)  #use the testing_env here to have the save_path available in the evaluation

    # evaluate_training(agent_container, testing_env, lab_journal=None, render_mode = 'flightgear')    #run the standardized test on the test_env
    evaluate_training(agent_container, testing_env, lab_journal=lab_journal)    #run the standardized test on the test_env
    
    exit(0)

    training_env = setup_env(arglist)
    testing_env = setup_env(arglist)

    #apply Varyetpoints to the training to increase the variance of training data
    training_env = VarySetpointsWrapper(training_env, prp.roll_deg, (-30, 30), (10, 30), (5, 30), (0.05, 0.5))
    training_env = VarySetpointsWrapper(training_env, prp.flight_path_deg, (-10, -5.5), (10, 45), (5, 30), (0.05, 0.5))
    training_env = VarySetpointsWrapper(training_env, prp.sideslip_deg, (-2, 2), (10, 45), (5, 30), (0.05, 0.5))


    agent_container = setup_container(training_env.task_list, arglist)

    save_test_run(testing_env, agent_container, lab_journal, arglist)  #use the testing_env here to have the save_path available in the evaluation

    perform_training(training_env, testing_env, agent_container, lab_journal, arglist)
    
    training_env.close()
    testing_env.close()


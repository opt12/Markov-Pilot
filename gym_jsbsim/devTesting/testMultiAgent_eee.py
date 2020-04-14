import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import argparse
import time
import pickle
import numpy as np

from gym_jsbsim.agent_task_eee import SingleChannel_FlightAgentTask
from gym_jsbsim.agents.pidAgent_eee import PID_Agent, PidParameters
from gym_jsbsim.environment_eee import NoFGJsbSimEnv_multi_agent
from gym_jsbsim.wrappers.episodePlotterWrapper_eee import EpisodePlotterWrapper_multi_agent
import gym_jsbsim.properties as prp

from gym_jsbsim.learn.evaluate_training_eee import test_net

from gym_jsbsim.reward_funcs_eee import make_glide_angle_reward_components, make_roll_angle_reward_components

## define the initial conditions TODO: should go into arglist
initial_path_angle_gamma_deg = 0
initial_roll_angle_phi_deg   = 0
initial_fwd_speed_KAS        = 95
initial_aoa_deg              = 1.0

def parse_args():   #TODO: adapt this. Taken from https://github.com/openai/maddpg/
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    # parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len-sec", type=int, default=120, help="maximum episode length in seconds (steps = seconds*interaction frequ.)")
    parser.add_argument("--num-episodes", type=int, default=100, help="number of episodes to train on")
    # parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    # parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    # parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--interaction-frequency", type=float, default=5, help="frequency of agent interactions with the environment")
    # Core training parameters
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="learning rate for the actor training Adam optimizer")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate for the critic training Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="number of episodes to optimize at the same time")
    parser.add_argument("--replay-size", type=int, default=1000000, help="size of the replay buffer")
    # parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='Default_Experiment', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    # parser.add_argument("--restore", action="store_true", default=False)
    # parser.add_argument("--display", action="store_true", default=False)
    # parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--testing-iters", type=int, default=2000, help="number of steps before running a performance test")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def setup_env(arglist) -> NoFGJsbSimEnv_multi_agent:
    agent_interaction_freq = arglist.interaction_frequency
    episode_time_s=arglist.max_episode_len_sec

    elevator_AT = SingleChannel_FlightAgentTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: -6.5},
                                make_base_reward_components= make_glide_angle_reward_components)

    aileron_AT = SingleChannel_FlightAgentTask('aileron', prp.aileron_cmd, {prp.roll_deg: -15}, max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_reward_components)

    agent_task_list = [elevator_AT, aileron_AT]
    
    env = NoFGJsbSimEnv_multi_agent(agent_task_list, agent_interaction_freq = agent_interaction_freq, episode_time_s = episode_time_s)
    env = EpisodePlotterWrapper_multi_agent(env)

    env.set_initial_conditions({ prp.initial_u_fps: 1.6878099110965*initial_fwd_speed_KAS
                                    , prp.initial_flight_path_deg: initial_path_angle_gamma_deg
                                    , prp.initial_roll_deg: initial_roll_angle_phi_deg
                                    , prp.initial_aoa_deg: initial_aoa_deg
                                    }) #just an example, sane defaults are already set in env.__init()__ constructor
    
    return env

def get_trainers(env, arglist):
    trainers = []
    agent_tasks = env.get_task_list()

    for at in agent_tasks:
        if at.name == 'elevator':
            elevator_pid_params = PidParameters( -5e-2, -6.5e-2, -1e-3)
            pid_elevator_agent = PID_Agent('elevator', elevator_pid_params, at.get_action_space(), agent_interaction_freq = arglist.interaction_frequency)
            trainers.append(pid_elevator_agent)
        if at.name == 'aileron':
            aileron_pid_params = PidParameters(3.5e-2,    1e-2,   0.0)
            pid_aileron_agent = PID_Agent('aileron', aileron_pid_params, at.get_action_space(), agent_interaction_freq = arglist.interaction_frequency)
            trainers.append(pid_aileron_agent)
    
    if len(trainers) != len(agent_tasks):
        raise LookupError('there must be an agent for each and every Agent_Task in the environment')

    return trainers
    

def train(arglist):

    # # Load previous results, if necessary #TODO: need to add some save/restore code compatible to pytorch
    # if arglist.load_dir == "":
    #     arglist.load_dir = arglist.save_dir
    # if arglist.display or arglist.restore or arglist.benchmark:
    #     print('Loading previous state...')
    #TODO: implement this correctly in PyTorch
    #     U.load_state(arglist.load_dir)

    max_episode_steps = arglist.max_episode_len_sec * arglist.interaction_frequency

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(len(trainers))]  # individual agent reward
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
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, _ = training_env.step(action_n)   #no need to process info_n
        episode_step += 1
        done = any(done_n)  #we end the episode if any of the involved agents came to an end
        terminal = training_env.is_terminal()   #there may be agent independent terminal conditions like the number of episode steps
        # collect experience, store to per-agent-replay buffers
        [agent.process_experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal) for i, agent in enumerate(trainers)]
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):         #there should be some np-magic to convert to a one liner
            episode_rewards[-1] += rew      #overall reward as sum of all agent rewards
            agent_rewards[i][-1] += rew

        if done or terminal:        #episode is over
            episode_counter += 1
            obs_n = training_env.reset()    #start new episode
            
            showPlot = False    #this is here for debugging purposes
            training_env.showNextPlot(show = showPlot)

            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)

        # increment global step counter
        train_step += 1

        # for benchmarking learned policies run the current agents on the testing_env
        if train_step % arglist.testing_iters == 0:
            testing_env.set_meta_information(episode_number = episode_counter)
            testing_env.showNextPlot(True)
            test_net(trainers, testing_env, add_exploration_noise=False)    #run the standardized test on the test_env

        # env.render(mode='flightgear') #not really useful in training

        # update all trainers
        loss = None #loss in the original code returns return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
        for agent in trainers:
            agent.preupdate()
        for agent in trainers:
            loss = agent.update(trainers, train_step)   #TODO: what is this good for?

        # save model, display training output   
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):    #save every arglist.save_rate completed episodes    
            # U.save_state(arglist.save_dir, saver=saver)
            # # print statement depends on whether or not there are adversaries
            # if num_adversaries == 0:
            #     print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
            #         train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
            # else:
            #     print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
            #         train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
            #         [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
            # t_start = time.time()

            # Keep track of final episode reward    TODO: check which outputs are really needed
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > arglist.num_episodes:
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
    training_env = setup_env(arglist)

    testing_env = setup_env(arglist)
    trainers = get_trainers(training_env, arglist)
    train(arglist)
    
    training_env.close()
    testing_env.close()


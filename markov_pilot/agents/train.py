import time
import argparse
import numpy as np
import pickle

from markov_pilot.environment.environment_eee import NoFGJsbSimEnv_multi_agent, JsbSimEnv_multi_agent
from markov_pilot.agents import AgentContainer
from markov_pilot.helper.lab_journal import LabJournal
from markov_pilot.testbed.evaluate_training_eee import evaluate_training


def perform_training(training_env: JsbSimEnv_multi_agent, testing_env: JsbSimEnv_multi_agent, agent_container: AgentContainer, lab_journal: LabJournal, arglist: argparse.Namespace):

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

    add_exploration_noise=True
    
    print('Starting iterations...')
    while True:
        # get action

        actions_n = agent_container.get_action(obs_n, add_exploration_noise=add_exploration_noise)
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


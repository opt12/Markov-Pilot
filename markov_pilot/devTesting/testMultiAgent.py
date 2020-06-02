import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

from markov_pilot.agent_task_eee import SingleChannel_FlightTask
from markov_pilot.agents.pidAgent_eee import PID_Agent, PidParameters
from markov_pilot.environment_eee import NoFGJsbSimEnv_multi
from markov_pilot.wrappers.episodePlotterWrapper_eee import EpisodePlotterWrapper_multi
import markov_pilot.properties as prp

from markov_pilot.reward_funcs_eee import make_glide_angle_reward_components, make_roll_angle_reward_components

if __name__ == '__main__':

    agent_interaction_freq = 5

    pid_elevator_AT = SingleChannel_FlightTask('elevator', prp.elevator_cmd, {prp.flight_path_deg: -6.5},
                                make_base_reward_components= make_glide_angle_reward_components)
    elevator_pid_params = PidParameters( -5e-2, -6.5e-2, -1e-3)
    pid_elevator_agent = PID_Agent('elevator', elevator_pid_params, pid_elevator_AT.action_space, agent_interaction_freq = agent_interaction_freq)

    pid_aileron_AT = SingleChannel_FlightTask('aileron', prp.aileron_cmd, {prp.roll_deg: -15}, max_allowed_error= 60, 
                                make_base_reward_components= make_roll_angle_reward_components)
    aileron_pid_params = PidParameters(3.5e-2,    1e-2,   0.0)
    pid_aileron_agent = PID_Agent('aileron', aileron_pid_params, pid_aileron_AT.action_space, agent_interaction_freq = agent_interaction_freq)

    agent_task_list = [pid_elevator_AT, pid_aileron_AT]
    trainers = [pid_elevator_agent, pid_aileron_agent]

    env = NoFGJsbSimEnv_multi(agent_task_list, agent_interaction_freq = agent_interaction_freq, episode_time_s=120)
    env = EpisodePlotterWrapper_multi(env)

    
    env.set_initial_conditions({prp.initial_flight_path_deg: -1.5}) #just an example, sane defaults are already set in env.__init()__ constructor
    
    obs_n = env.reset()
    pid_elevator_agent.reset_notifier() #only needed for the PID_Agent as it maintains internal state
    pid_aileron_agent.reset_notifier()  #only needed for the PID_Agent as it maintains internal state

    episode_step = 0

    env.showNextPlot(show = True)

    
    while True:
        # get action
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        #env.render(mode='flightgear')
        episode_step += 1
        done = any(done_n)
        terminal = env.is_terminal()

        if episode_step%150 == 0:
            env.change_setpoints({prp.roll_deg: -env.sim[prp.roll_deg]})

        if episode_step%180 == 0:
            env.change_setpoints({prp.flight_path_deg: env.sim[prp.flight_path_deg]+0.5})

        # collect experience
        # for i, agent in enumerate(trainers):
        #     agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
        obs_n = new_obs_n

        if done or terminal:
            break
    
    print(f"Episode ended after {episode_step} steps.")
    print(f"env.state after the Episode: {env.state}")


    


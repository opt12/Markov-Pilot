import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!
import time
import gym

import gym_jsbsim
from gym_jsbsim.agents import PIDAgent, RandomAgent, ConstantAgent
from gym_jsbsim.wrappers.episodePlotterWrapper import EpisodePlotterWrapper
from gym_jsbsim.wrappers import PidWrapper, PidWrapperParams, PidParameters
import gym_jsbsim.properties as prp

if __name__ == "__main__":
    env = gym.make("JSBSim-SteadyGlideTask-Cessna172P-Shaping.STANDARD-NoFG-v0")
    # env = gym.make("JSBSim-SteadyRollAngleTask-Cessna172P-Shaping.STANDARD-FG-v0")
    env = EpisodePlotterWrapper(env)    #to show a summary of the next epsode, set env.showNextPlot(True)

    # env_F15 = gym.make("JSBSim-SteadyRollAngleTask-F15-Shaping.STANDARD-FG-v0")
    # env_F15 = EpisodePlotterWrapper(env_F15)    #to show a summary of the next epsode, set env.showNextPlot(True)



    # elevator params: 'Kp':  -5e-2, 'Ki': -6.5e-2, 'Kd': -1e-3
    # aileron prams:   'Kp': 3.5e-2, 'Ki':    1e-2, 'Kd': 0.0
    elevator_wrap = PidWrapperParams('fcs_elevator_cmd_norm', 'error_glideAngle_error_deg', PidParameters( -5e-2, -6.5e-2, -1e-3))
    aileron_wrap  = PidWrapperParams('fcs_aileron_cmd_norm',  'error_rollAngle_error_deg',  PidParameters(3.5e-2,    1e-2,   0.0))
    # env = PidWrapper(env, [elevator_wrap, aileron_wrap])

    env = EpisodePlotterWrapper(env)    #to show a summary of the next epsode, set env.showNextPlot(True)
    env = PidWrapper(env, [aileron_wrap, elevator_wrap])
    env.task.change_setpoints(env.sim, { prp.setpoint_flight_path_deg: -5.7
                        , prp.setpoint_roll_angle_deg: -0
                        , prp.episode_steps: 3000})
    env.task.set_initial_ac_attitude(-6, -0, 95)

    # env_F15 = PidWrapper(env_F15, [aileron_wrap, elevator_wrap])

    # agent_F15 = PIDAgent(env_F15.action_space, agent_interaction_freq, env_F15)
    agent_interaction_freq = env.JSBSIM_DT_HZ / env.sim_steps_per_agent_step
    # agent = PIDAgent(env.action_space, agent_interaction_freq, env)
    agent = RandomAgent(env.action_space)
    

    for i in range(1):
        state = env.reset()
        total_reward = 0.0
        # state_F15 = env_F15.reset()
        total_reward_F15 = 0.0

        frame_idx = 0
        ts_frame = 0
        ts = time.time()
        if i%10 == 0:
            env.showNextPlot(True, True)
            # env_F15.showNextPlot(True, True)

        while True: # this is what was originally done in the tensorforce runner
            frame_idx += 1
            action = agent.act(state)
            state, reward, done, _ = env.step(0.05*action)

            # action_F15 = agent_F15.act(state_F15)
            # state_F15, reward_F15, done_F15, _ = env_F15.step(action_F15)
            # env.render('human')
            # env.render('timeline')
            # env_F15.render('timeline')
            # env.render('flightgear')
            total_reward += reward
            # total_reward_F15 += reward_F15
            # if frame_idx%10 == 0:
            #     #just in case we want some progress feedback
            #     print(frame_idx)
            if frame_idx == 299:
                env.task.change_setpoints(env.sim, { 
                      prp.setpoint_roll_angle_deg: -10
                })
            if frame_idx == 599:
                env.task.change_setpoints(env.sim, { prp.setpoint_roll_angle_deg: 10
                })

            if done: # or 'done_F15'
                break
        runtime = (time.time() - ts)
        speed = (frame_idx - ts_frame) / runtime
        print("%d frames in %f seconds: %f fps"% (frame_idx, runtime, speed))
        print("Cessna: Reward got: %.2f" % total_reward)
        # print("F15: Reward got: %.2f" % total_reward_F15)


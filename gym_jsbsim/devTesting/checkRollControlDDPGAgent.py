import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!
import time
import gym
import numpy as np

import gym_jsbsim
from gym_jsbsim.agents import PIDAgent, RandomAgent, ConstantAgent
from episodePlotterWrapper import EpisodePlotterWrapper

if __name__ == "__main__":
    env = gym.make("JSBSim-SteadyRollAngleTask-Cessna172P-Shaping.STANDARD-NoFG-v0")
    env = EpisodePlotterWrapper(env)    #to show a summary of the next epsode, set env.showNextPlot(True)

    agent_interaction_freq = env.JSBSIM_DT_HZ / env.sim_steps_per_agent_step
    elevatorAgent = PIDAgent(env.action_space, agent_interaction_freq, env)
    aileronAgent  = RandomAgent(env.action_space)
    

    for i in range(1):
        state = env.reset()
        total_reward = 0.0
        frame_idx = 0
        ts_frame = 0
        ts = time.time()
        if i%10 == 0:
            env.showNextPlot(True, False)

        while True: # this is what was originally done in the tensorforce runner
            frame_idx += 1
            elevatorAction, _ = elevatorAgent.act(state)
            _, aileronAction  = aileronAgent.act(state)
            action = np.array([elevatorAction, aileronAction])
            state, reward, done, _ = env.step(action)
            # env.render('human')
            # env.render('timeline')
            # env.render('flightgear')
            total_reward += reward
            # if frame_idx%10 == 0:
            #     #just in case we want some progress feedback
            #     print(frame_idx)
            if done:
                break
        runtime = (time.time() - ts)
        speed = (frame_idx - ts_frame) / runtime
        print("%d frames in %f seconds: %f fps"% (frame_idx, runtime, speed))
        print("Reward got: %.2f" % total_reward)

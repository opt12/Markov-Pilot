import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/')

import gym
import gym_jsbsim
import numpy as np

import random
import time


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        #implicit else
        return action


if __name__ == "__main__":
    env = gym.make("JSBSim-SteadyGlideTask-Cessna172P-Shaping.STANDARD-NoFG-v0")
    # env = gym.make("JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0")

    obs = env.reset()
    total_reward = 0.0

    frame_idx = 0
    ts_frame = 0
    ts = time.time()

    while True:
        frame_idx += 1
#        action = env.action_space.sample()
        action = env.action_space.sample()
        # action = np.array([0,0])
        obs, reward, done, _ = env.step(action)
        env.render('human')
        # env.render('timeline')
        # env.render('flightgear')
        # print("Action: {}; Observation: {}".format(action, obs))
        total_reward += reward
        if done:
            break
    runtime = (time.time() - ts)
    speed = (frame_idx - ts_frame) / runtime
    print("%d frames in %f seconds: %f fps"% (frame_idx, runtime, speed))

    print("Reward got: %.2f" % total_reward)

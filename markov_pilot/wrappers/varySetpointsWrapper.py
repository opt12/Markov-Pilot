#!/usr/bin/env python3
import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import gym
import numpy as np
import math
import random
import markov_pilot.environment.properties as prp

from abc import ABC, abstractmethod

from markov_pilot.environment.properties import BoundedProperty
from typing import Tuple, List

class VarySetpointsWrapper(gym.Wrapper):
    """
    A wrapper to vary the setpoints at the beginning of each episode

    This can be used during training to have bigger variance in the training data
    """

    class SetpointVariator(ABC):
        """
        A helper that can vary a setpoint between two extreme values following a specific pattern
        """
        @abstractmethod
        def vary(self):
            ''' outputs the setpoint for the next step or None if there is nothing to do'''
            ...
        @abstractmethod
        def start_variation(self):
            ''' starts the setpoint variation for the first time in the upcoming interval
            :return: the first setpoint to be passed to the env'''
            ...

    class StepVariator(SetpointVariator):
        def __init__(self, setpoint_range):
            self.min = setpoint_range[0]
            self.max = setpoint_range[1]
        def vary(self):
            return None #for a step function, there is no more to do after the first step
        def start_variation(self, _):
            return random.uniform(self.min, self.max)

    class RampVariator(SetpointVariator):
        def __init__(self, setpoint_range, ramp_time, dt):
            self.min = setpoint_range[0]
            self.max = setpoint_range[1]
            self.t_min = ramp_time[0]
            self.t_max = ramp_time[1]
            self.dt = dt            #the time interval between two subsequent calls
        def vary(self):
            if self.steps_left >0:
                self.current_value += self.delta
                self.steps_left -=1
                return self.current_value
            else:
                return None
        def start_variation(self, current_value):
            self.target_value = random.uniform(self.min, self.max)
            ramp_length = random.uniform(self.t_min, self.t_max)
            self.steps_left = ramp_length / self.dt
            self.delta = (self.target_value - current_value)/self.steps_left
            self.current_value = current_value
            return self.vary()

    class SineVariator(SetpointVariator):
        def __init__(self, setpoint_range, sine_frequ, dt):
            self.min = setpoint_range[0]
            self.max = setpoint_range[1]
            self.freq_min = sine_frequ[0]
            self.freq_max = sine_frequ[1]
            self.dt = dt
        def vary(self):
            self.step += 1
            return self.mean_value + math.sin(self.step * self.sine_increment)*self.amplitude
        def start_variation(self, current_value):
            frequ = random.uniform(self.freq_min, self.freq_max)
            self.sine_increment = self.dt* frequ * 2*math.pi    #the increment in the sine argument within dt

            self.mean_value = current_value
            if self.min > self.mean_value or self.mean_value > self.max:
                self.mean_value = random.uniform(self.min, self.max)  #this may happen due to unfortunate initial conditions

            max_amplitude = min(self.mean_value - self.min, self.max - self.mean_value)
            self.amplitude = random.uniform(0, max_amplitude) 
            self.amplitude *= random.randrange(-1,2, 2)  #change the direction

            self.step = 0
            return self.vary()

    def __init__(self, env, setpoint_property: BoundedProperty, 
                setpoint_range: Tuple[float, float], 
                interval_length: Tuple[float, float] = (5., 120.),
                ramp_time:Tuple[float, float] = (0,0), sine_frequ: Tuple[float, float] = (0,0), 
                initial_conditions: List[Tuple] = []):
        """
        :param setpoint_property: The property which describes the setpoint. 
        :param setpoint_range: the range the setpoint may be chosen from (min, max)
        :param interval_length: the time in seconds for the interval till the next change
        :param ramp_time: the time, a ramp may last from current setpoint to target setpoint; (0, 0) disables ramps
        :param sine_frequ: the frequqcy range from which sine modulation may be chosen; (0,0) disables sine modulation
        :param initial_conditions: TODO: specify the initial conditions that may be varied and their ranges.
        """
        self.env = env
        self.setpoint_property = setpoint_property
        self.setpoint_range = setpoint_range
        self.interval_length = interval_length
        self.ramp_time = ramp_time
        self.sine_frequ = sine_frequ
        self.initial_conditions = initial_conditions

        #don't restore the VarySetpoints wrapper automatically
        # #append the restore data
        # self.env_init_dicts.append({
        #     'setpoint_property': setpoint_property,
        #     'setpoint_range': setpoint_range,
        #     'interval_length': interval_length,
        #     'ramp_time': ramp_time,
        #     'sine_frequ': sine_frequ,
        #     'initial_conditions': initial_conditions,
        # })
        # self.env_classes.append(self.__class__.__name__)

        step_variator = self.StepVariator(setpoint_range)
        ramp_variator = self.RampVariator(setpoint_range, ramp_time, self.dt)
        sine_variator = self.SineVariator(setpoint_range, sine_frequ, self.dt)
        
        self.enabled_variators = [step_variator]
        if ramp_time != (0, 0): self.enabled_variators.append(ramp_variator)
        if sine_frequ != (0, 0): self.enabled_variators.append(sine_variator) 

        self.envs_to_vary = [self.env]

    def inject_other_env(self, env):
        '''if the setpoint changes shall affect more than one environment synchronously; e. g. for benchmarking'''
        self.envs_to_vary.append(env)

    def _initialize_next_variation(self):
        interval = random.uniform(self.interval_length[0], self.interval_length[1])
        self.steps_till_next_variation = int(interval / self.dt)
        variator_idx = random.randrange(0, len(self.enabled_variators))
        self.active_variator = self.enabled_variators[variator_idx]
        current_value = self.env.sim[self.setpoint_property]
        return self.active_variator.start_variation(current_value)

    def step(self, action):
        # pylint: disable=method-hidden
        if not self.steps_till_next_variation:
            varied_setpoint = self._initialize_next_variation()
        else:
            varied_setpoint = self.active_variator.vary()
        self.steps_till_next_variation -= 1
        if varied_setpoint: [env.change_setpoints({self.setpoint_property: varied_setpoint}) for env in self.envs_to_vary]
        return self.env.step(action)

    def reset(self):
        # pylint: disable=method-hidden
        varied_setpoint = self._initialize_next_variation()
        [env.change_setpoints({self.setpoint_property: varied_setpoint}) for env in self.envs_to_vary]

        #TODO: here goes the modfication of the initial conditions
        # not now [self.envs_to_vary.set_initial_conditions( {})]
        
        return self.env.reset()
    

    




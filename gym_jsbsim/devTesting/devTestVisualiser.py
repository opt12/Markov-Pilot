import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim/')
import time

import types
from collections import namedtuple
print(sys.path)
import gym_jsbsim.utils
from gym_jsbsim.properties import BoundedProperty, Property
from typing import Optional, Sequence, Dict, Tuple, NamedTuple, Type
from gym_jsbsim.simulation import Simulation
import math

from gym_jsbsim.tests.stubs import SimStub
from gym_jsbsim.visualiser import TimeLineVisualiser
import gym_jsbsim.properties as prp


def get_props_to_output() -> Tuple:
    return (prp.u_fps,)# prp.altitude_sl_ft, prp.roll_rad, prp.flight_path_deg,)

def updateFakeData(sim: Simulation) -> None:
    updateFakeData.time +=1
    val_sin = math.sin(updateFakeData.time)
    val_cos = math.cos(updateFakeData.time)
    for prop in get_props_to_output():
        sim[prop] = (prop.max - prop.min) / 2 * val_sin + (prop.max + prop.min) / 2
    for prop in [prp.aileron_left, prp.elevator, prp.throttle, prp.rudder]:
        sim[prop] = (prop.max - prop.min) / 2 * val_sin + (prop.max + prop.min) / 2
    for prop in [prp.aileron_cmd, prp.elevator_cmd, prp.throttle_cmd, prp.rudder_cmd]:
        sim[prop] = (prop.max - prop.min) / 2 * val_cos + (prop.max + prop.min) / 2
    return

updateFakeData.time =0  #this is kind of a static variable https://stackoverflow.com/a/279586/2682209


if __name__ == '__main__':
    sim = SimStub()
    updateFakeData(sim)
    props = get_props_to_output()
    figure_visualiser = TimeLineVisualiser(sim, props)  #simulation parameter is not used

    figure_visualiser.plot(sim)
    tstart = time.time()
    for i in range(500):
        # print(i)
        updateFakeData(sim)
        # https://bastibe.de/2013-05-30-speeding-up-matplotlib.html
        figure_visualiser.plot(sim)
    tend = time.time()
    print("FPS: %.2f"%(500./(tend-tstart)))

    # import matplotlib.pyplot as plt
    # import matplotlib as mpl
    # import numpy as np

    # print ("Backend: " + mpl.get_backend())

    # x = np.linspace(0, 20, 100)
    # plt.plot(x, np.sin(x))
    # plt.show()







import functools
import operator
import numpy as np

from typing import Tuple, List
from typing import Dict, Iterable
from gym.spaces import Box

from markov_pilot.environment.aircraft import cessna172P, a320, f15


def aggregate_gym_boxes(ac_spcs_n: List[Box]) -> Box:
    """
    :return: the combined action space from the input list
    """
    lows_n = list(map(lambda bx: bx.low, ac_spcs_n))
    highs_n = list(map(lambda bx: bx.high, ac_spcs_n))

    return Box(np.concatenate(lows_n), np.concatenate(highs_n))

def box2dict(space):    #only supports 1 dimensional 
    return {
                'shape': space.shape, 
                'low': list(space.low),
                'high': list(space.high),
            }

def dict2Box(boxdict: Dict) -> Box:
    return Box(boxdict['low'], boxdict['high'])

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class AttributeFormatter(object):
    """
    Replaces characters that would be illegal in an attribute name

    Used through its static method, translate()
    """
    ILLEGAL_CHARS = '\-/.'
    TRANSLATE_TO = '_' * len(ILLEGAL_CHARS)
    TRANSLATION_TABLE = str.maketrans(ILLEGAL_CHARS, TRANSLATE_TO)

    @staticmethod
    def translate(string: str):
        return string.translate(AttributeFormatter.TRANSLATION_TABLE)


def get_env_id(task_type, aircraft, shaping, enable_flightgear) -> str:
    """
    Creates an env ID from the environment's components

    :param task_type: Task class, the environment's task
    :param aircraft: Aircraft namedtuple, the aircraft to be flown
    :param shaping: HeadingControlTask.Shaping enum, the reward shaping setting
    :param enable_flightgear: True if FlightGear simulator is enabled for visualisation else False
     """
    if enable_flightgear:
        fg_setting = 'FG'
    else:
        fg_setting = 'NoFG'
    return f'JSBSim-{task_type.__name__}-{aircraft.name}-{shaping}-{fg_setting}-v0'


def get_env_id_kwargs_map() -> Dict[str, Tuple]:
    """ Returns all environment IDs mapped to tuple of (task, aircraft, shaping, flightgear) """
    # lazy import to avoid circular dependencies
    from markov_pilot.tasks_old import Shaping, HeadingControlTask, TurnHeadingControlTask
    from markov_pilot.task_steadyGlide import SteadyRollGlideTask, SteadyRollAngleTask, SteadyGlideAngleTask

    map = {}
    for task_type in (HeadingControlTask, TurnHeadingControlTask, SteadyRollGlideTask, SteadyRollAngleTask, SteadyGlideAngleTask):
        for plane in (cessna172P, a320, f15):
            for shaping in (Shaping.STANDARD, Shaping.EXTRA, Shaping.EXTRA_SEQUENTIAL):
                for enable_flightgear in (True, False):
                    id = get_env_id(task_type, plane, shaping, enable_flightgear)
                    assert id not in map
                    map[id] = (task_type, plane, shaping, enable_flightgear)
    return map


def product(iterable: Iterable):
    """
    Multiplies all elements of iterable and returns result

    ATTRIBUTION: code provided by Raymond Hettinger on SO
    https://stackoverflow.com/questions/595374/whats-the-function-like-sum-but-for-multiplication-product
    """
    return functools.reduce(operator.mul, iterable, 1)


def reduce_reflex_angle_deg(angle: float) -> float:
    """ Given an angle in degrees, normalises in [-179, 180] """
    # ATTRIBUTION: solution from James Polk on SO,
    # https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees#
    new_angle = angle % 360
    if new_angle > 180:
        new_angle -= 360
    return new_angle

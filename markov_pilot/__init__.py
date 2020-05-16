#like seen in https://stackoverflow.com/a/8663557/2682209

import sys, os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'markov_pilot'))

import markov_pilot

del sys.path[0], sys, os

# import gym.envs.registration
# import enum
# # from markov_pilot.tasks_old import Task, HeadingControlTask, TurnHeadingControlTask
# # from markov_pilot.task_steadyGlide import SteadyRollGlideTask, SteadyRollAngleTask, SteadyGlideAngleTask
# from markov_pilot.environment.aircraft import Aircraft, cessna172P

# import environment
# import helper
# import agents
# import tasks
# import wrappers

# from markov_pilot.helper import utils


# """
# This script registers all combinations of task, aircraft, shaping settings
#  etc. with OpenAI Gym so that they can be instantiated with a gym.make(id)
#  command.

# The markov_pilot.Envs enum stores all registered environments as members with
#  their gym id string as value. This allows convenient autocompletion and value
#  safety. To use do:
#        env = gym.make(markov_pilot.Envs.desired_environment.value)
# """

# # for env_id, (task, plane, shaping, enable_flightgear) in utils.get_env_id_kwargs_map().items():
# #     if enable_flightgear:
# #         entry_point = 'markov_pilot.environment:JsbSimEnv'
# #     else:
# #         entry_point = 'markov_pilot.environment:NoFGJsbSimEnv'
# #     kwargs = dict(task_type=task,
# #                   aircraft=plane,
# #                   shaping=shaping)
# #     gym.envs.registration.register(id=env_id,
# #                                    entry_point=entry_point,
# #                                    kwargs=kwargs)
# #     # print("registered: {}".format(env_id))    #prints out all registered environments to copy the name from

# # # make an Enum storing every Gym-JSBSim environment ID for convenience and value safety
# # Envs = enum.Enum.__call__('Envs', [(utils.AttributeFormatter.translate(env_id), env_id)
# #                                    for env_id in utils.get_env_id_kwargs_map().keys()])

import sys
import os
import pickle
import importlib
from typing import Union, List

from markov_pilot.environment.environment import JsbSimEnv_multi, NoFGJsbSimEnv_multi
from markov_pilot.agents.agent_container import AgentContainer
from markov_pilot.helper.lab_journal import LabJournal

#these imports are needed to restore all the classes from the saved pickle-files
from markov_pilot.tasks.tasks import SingleChannel_FlightTask, SingleChannel_MinimumProps_Task
from markov_pilot.wrappers.episodePlotterWrapper import EpisodePlotterWrapper_multi



def save_test_run(env: JsbSimEnv_multi, agent_container: AgentContainer, lab_journal: LabJournal, arglist):
    """
    - creates a suitable directory for the test run
    - adds a sidecar file containing the meta information on the run (dict saved as pickle)
    - adds a text file containing the meta information on the run
    - add a line to the global csv-file for the test run
    """
    # IMPORTANT to do this first,  to make the lab_journal aware of the start time of the run
    lab_journal.set_run_start()

    task_names = '_'.join([t.name for t in env.task_list])
    date = lab_journal.run_start.strftime("%Y_%m_%d")
    time = lab_journal.run_start.strftime("%H-%M")

    #build the path name for the run protocol
    save_path = os.path.join(lab_journal.journal_save_dir, env.aircraft.name, arglist.exp_name, task_names, date+'-'+time)

    #create the base directory for this test_run
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #create the directories for each agent_task
    for a in agent_container.agents_m:
        agent_path = os.path.join(save_path, a.name)
        os.makedirs(os.path.join(save_path, a.name), exist_ok=True)
        a.set_save_path(agent_path)

    env.save_env_data(arglist, save_path)
    agent_container.save_agent_container_data(save_path)
    #eventually append the run data to the csv-file. 
    csv_line_nr = lab_journal.append_run_data(env, agent_container.agents_m, save_path)
    env.set_meta_information(csv_line_nr = csv_line_nr)

def restore_env_from_journal(lab_journal, line_numbers: Union[int, List[int]], target_environment = None) -> NoFGJsbSimEnv_multi:
    ENV_PICKLE = 'environment_init.pickle'  #these are hard default names for the files
    TASKS_PICKLE ='task_agent.pickle'       #these are hard default names for the files

    ln = line_numbers if isinstance(line_numbers, int) else line_numbers[0]

    #get run protocol
    try:
        model_file = lab_journal.get_model_filename(ln)
        run_protocol_path = lab_journal.find_associated_run_path(model_file)
    except TypeError:
        print(f"there was no run protocol found that is associated with line_number {ln}")
        exit()

    if run_protocol_path == None:
        print(f"there was no run protocol found that is associated with line_number {ln}")
        exit()

    #load the TASKS_PICKLE and restore the task_list
    with open(os.path.join(run_protocol_path, TASKS_PICKLE), 'rb') as infile:
        task_agent_data = pickle.load(infile)
    
    task_agents = []
    for idx in range(len(task_agent_data['task_list_class_names'])):
        task_list_init = task_agent_data['task_list_init'][idx]
        task_list_class_name = task_agent_data['task_list_class_names'][idx]
        make_base_reward_components_file = task_agent_data['make_base_reward_components_file'][idx]
        make_base_reward_components_fn = task_agent_data['make_base_reward_components_fn'][idx]

        #load the make_base_reward_components function from a python-file
        #load function from given filepath  https://stackoverflow.com/a/67692/2682209
        spec = importlib.util.spec_from_file_location("make_base_rwd", os.path.join(run_protocol_path, make_base_reward_components_file))
        func_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(func_module)
        make_base_reward_components = getattr(func_module, make_base_reward_components_fn)

        #get the class for the task_agent https://stackoverflow.com/a/17960039/2682209
        class_ = getattr(sys.modules[__name__], task_list_class_name)
        #add the make_base_reward_components function to the parameter dict
        task_list_init.update({'make_base_reward_components': make_base_reward_components})
        #transform the setpoint_props and setpoint_values lists to setpoints dict
        task_list_init.update({'setpoints': dict(zip(task_list_init['setpoint_props'], task_list_init['setpoint_values']))})
        del task_list_init['setpoint_props']
        del task_list_init['setpoint_values']
        ta = class_(**task_list_init)
        task_agents.append(ta)

    #the task_agents are now ready, so now let's prepare the environment
    #load the ENV_PICKLE and restore the task_list
    with open(os.path.join(run_protocol_path, ENV_PICKLE), 'rb') as infile:
        env_data = pickle.load(infile)

    env_init_dicts = env_data['init_dicts']
    env_classes = env_data['env_classes']

    #create the innermost environment with the task_list added
    #load the env class
    env_class_ = getattr(sys.modules[__name__], env_classes[0])

    if target_environment and (env_class_ == JsbSimEnv_multi or env_class_ == NoFGJsbSimEnv_multi):
        #the user wants us to exchange the innermost environment
        if target_environment == 'NoFG':
            env_class_ = NoFGJsbSimEnv_multi
        elif target_environment == 'FG':
            env_class_ = JsbSimEnv_multi
        else:
            raise ValueError("parameter target_:environment must be either 'NoFG' or 'FG' or entirely omitted. Other values not allowed.")

    env_init = env_init_dicts[0]
    env_init.update({'task_list':task_agents})

    env = env_class_(**env_init)
    
    #apply wrappers if available
    #TODO: what about other wrappers than EpisodePlotterWrapper? We don't save the VarySetpointWrapper to the wrappers list.
    for idx in range(1, len(env_init_dicts)):
        wrapper_class_ = getattr(sys.modules[__name__], env_classes[idx])
        wrap_init = env_init_dicts[idx]
        wrap_init.update({'env': env})
        env = wrapper_class_(**wrap_init)
    
    env.set_meta_information(csv_line_nr = ln)  #set the line number, the environment was loaded from
    return env

def restore_agent_container_from_journal(lab_journal, line_numbers: Union[int, List[int]], task_list_n = None, mapping_dict = None) -> 'AgentContainer':
    CONTAINER_PICKLE = 'agent_container.pickle'

    ln = [line_numbers] if isinstance(line_numbers, int) else line_numbers

    #get run protocol

    agent_pickle_files_m = [lab_journal.get_model_filename(line) for line in ln]
    try:
        model_file = lab_journal.get_model_filename(ln[0])
        run_protocol_path = lab_journal.find_associated_run_path(model_file)
    except TypeError:
        print(f"there was no run protocol found that is associated with line_number {ln}")
        exit()

    agent_container = AgentContainer.init_from_save(os.path.join(run_protocol_path, CONTAINER_PICKLE), agent_pickle_files_m, task_list_n, mapping_dict)

    return agent_container

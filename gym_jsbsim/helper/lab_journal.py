import datetime
import os
import csv
import json
import datetime

class LabJournal():
    def __init__(self, base_dir, arglist):
        #check if lab journal file already exists
        self._jornal_save_dir = os.path.join(base_dir, 'testruns') 
        try:    #arglist is either given as arglist namespace or as dictionary
            self.arglist_dict = vars(arglist)
        except TypeError:
            self.arglist_dict = arglist if isinstance(arglist, dict) else {}   #if nor proper arglist was given

        os.makedirs(self._jornal_save_dir, exist_ok=True)
        self.journal_file_name = os.path.join(self._jornal_save_dir, 'lab_journal.csv')
        self.column_names = ['line_number', 'entry_type', 'reward', 'steps', 'date', 'time', 'path', 'agent_task_classes', 'trainer_classes'] + list(self.arglist_dict.keys())
        self.journal_entries = []
        
        csv_exists = os.path.isfile(self.journal_file_name)
        if csv_exists:
            #read in all testruns so far (I know this maybe a lot, but so what, we're doing big data)
            with open(self.journal_file_name, mode='r') as infile:
                reader = csv.DictReader(infile)
                self.journal_entries = list(reader)
                if len(self.journal_entries) > 0:
                    self.next_line_number = int(self.journal_entries[-1]['line_number']) + 1
                else:
                    self.next_line_number = 0
        else:
            with open(self.journal_file_name, 'a') as outfile:   # TODO: add some sensible information there
                outfile = csv.DictWriter(outfile, self.column_names)
                outfile.writeheader()
            self.next_line_number = 0
    
    @property
    def run_start(self):
        return self._run_start
    def set_run_start(self):
        self._run_start = datetime.datetime.now()   #update the time the saved run was started to get the directories right
        return self._run_start

    @property
    def journal_save_dir(self):
        return self._jornal_save_dir

    def _write_data(self, data_dict):
        data_dict.update({'line_number': self.next_line_number})
        with open(self.journal_file_name, 'a') as outfile:   # TODO: add some sensible information there
            outfile = csv.DictWriter(outfile, self.column_names)
            outfile.writerow(data_dict)

        self.journal_entries.append(data_dict)
        self.next_line_number += 1
        return self.next_line_number-1  #the line number we've just written to

    def append_run_data(self, env, agents, save_path) -> int:
        """
        :return: the line number in the CSV file that was just written
        """

        self.save_path = save_path

        #create a sidecar file containing the meta information of the test_run
        agents_classes_dict = {tr.name: tr.__class__.__name__ for tr in agents}
        agent_task_classes_dict = {at.name: at.__class__.__name__ for at in env.task_list}
        run_dict = {
            'entry_type': 'env_description',         #one of 'env_description' or agent_name
            'date': self._run_start.strftime("%d.%m.%Y"),
            'time': self._run_start.strftime("%H:%M:%S"),
            'path': 'file://'+os.path.abspath(self.save_path), 
            'reward': '',
            'steps': '',
            'trainer_classes': agents_classes_dict,
            'agent_task_classes': agent_task_classes_dict
        }
        run_dict.update(self.arglist_dict)

        return self._write_data(run_dict)
    
    def append_evaluation_data(self, eval_dict):
        now = datetime.datetime.now()
        name = eval_dict['entry_type']
        eval_dict.update({
            'date': now.strftime("%d.%m.%Y"),
            'time': now.strftime("%H:%M:%S"),
        })
        self._write_data(eval_dict)
    
    def _find_key_in_journal(self, line_number):
        """
        performs a binary search in self.journal_entries['line_number'] for a certain key
        and returns the associated index
        """ 
        def _center(range):
            return (range[0]+range[1])//2
        
        range = [0, len(self.journal_entries)-1]
        while range[0]<= range[1]:
            current_idx = _center(range)
            if int(self.journal_entries[current_idx]['line_number']) == line_number:
                return current_idx
            if int(self.journal_entries[current_idx]['line_number']) < line_number:
                range[0] = current_idx + 1
            else:
                range[1] = current_idx - 1
        #if we arrive here, line_:number was not found
        return None

    def find_associated_run_path(self, start_path):
        """
        starting form a given path, the file system is traversed upwards to find environment_data.json and agents_data.json.

        :param start_path: a file or folder name from where the file_system is traversed upwards

        :return: The path containing the associated environment_data.json and agents_data.json
        """
        ENV_FILE = 'environment_data.json'
        AG_FILE  = 'agent_container.json'

        #get folder name
        if (os.path.isdir(start_path)):
            folder_path = start_path
        elif (os.path.isfile(start_path)):
            folder_path = os.path.dirname(start_path)
        else:
            folder_path =''

        #check, if files are available
        while not (os.path.isfile(os.path.join(folder_path, ENV_FILE)) and 
                   os.path.isfile(os.path.join(folder_path, AG_FILE))):
            folder_path = os.path.dirname(folder_path)    #go one level up
            if folder_path == '':
                folder_path = None #we didn't find the data
                break
        
        return folder_path

    def get_model_filename(self, line_number):
        """
        fetches the model filename from a line of the lab journal

        :param line_number: the line number of the model to fetch. It's *not* the physical line
            number but the key in the first column of the lab journal dataset
        """

        #we assume the lab_journal file to be open and all data available in self.journal_entries
        idx = self._find_key_in_journal(line_number)
        file_name = None
        if idx:
            file_name =  self.journal_entries[idx].get('path', None)
            prefix = 'file://'
            file_name = file_name[file_name.startswith(prefix) and len(prefix):]
        return file_name


if __name__ == '__main__':

    import argparse
    no_arglist = argparse.Namespace()
    lab_journal = LabJournal("./", no_arglist)

    model_file = lab_journal.get_model_filename(666)
    print(f'model_file: {model_file}')

    model_file = lab_journal.get_model_filename(-1)
    print(f'model_file: {model_file}')
    model_file = lab_journal.get_model_filename(139)
    print(f'model_file: {model_file}')

    run_path = lab_journal.find_associated_run_path(model_file)
    print(f'run_path for {model_file}: {run_path}')

    model_file = lab_journal.get_model_filename(10)
    print(f'model_file: {model_file}')
    run_path = lab_journal.find_associated_run_path(model_file)
    print(f'run_path for {model_file}: {run_path}')

    model_file = lab_journal.get_model_filename(97)
    print(f'model_file: {model_file}')
    run_path = lab_journal.find_associated_run_path(model_file)
    print(f'run_path for {model_file}: {run_path}')

    

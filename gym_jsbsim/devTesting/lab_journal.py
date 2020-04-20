import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim-eee/') #TODO: Is this a good idea? Dunno! It works!

import datetime
import os
import csv
import json
import datetime

class LabJournal():
    def __init__(self, base_dir, arglist):
        #check if lab journal file already exists
        
        self.base_dir = base_dir

        self.journal_file_name = os.path.join(base_dir, 'testruns', 'lab_journal.csv')
        self.column_names = ['line_number', 'entry_type', 'reward', 'steps', 'date', 'time', 'path', 'agent_task_classes', 'trainer_classes'] + list(vars(arglist).keys())
        self.arglist = vars(arglist)
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
    
    def _write_data(self, data_dict):
        data_dict.update({'line_number': self.next_line_number})
        with open(self.journal_file_name, 'a') as outfile:   # TODO: add some sensible information there
            outfile = csv.DictWriter(outfile, self.column_names)
            outfile.writerow(data_dict)

        self.journal_entries.append(data_dict)
        self.next_line_number += 1
        

    def append_run_data(self, env, agents, run_start, save_path):
        self.save_path = save_path

        #create a sidecar file containing the meta information of the test_run
        agents_classes_dict = {tr.name: tr.__class__.__name__ for tr in agents}
        agent_task_classes_dict = {at.name: at.__class__.__name__ for at in env.task_list}
        run_dict = {
            'entry_type': 'env_description',         #one of 'env_description' or agent_name
            'date': run_start.strftime("%d.%m.%Y"),
            'time': run_start.strftime("%H:%M:%S"),
            'path': 'file://'+os.path.abspath(self.save_path), 
            'reward': '',
            'steps': '',
            'trainer_classes': agents_classes_dict,
            'agent_task_classes': agent_task_classes_dict
        }
        run_dict.update(self.arglist)

        self._write_data(run_dict)
    
    def append_evaluation_data(self, eval_dict):
        now = datetime.datetime.now()
        name = eval_dict['entry_type']
        eval_dict.update({
            'date': now.strftime("%d.%m.%Y"),
            'time': now.strftime("%H:%M:%S"),
            'path': 'file://'+os.path.abspath(self.save_path)+'/'+name,
        })
        self._write_data(eval_dict)
    

    

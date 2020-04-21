import gym
import gym.spaces
import gym_jsbsim.properties as prp
import numpy as np
import pandas as pd
import datetime
import os
import math
import shutil   #to have copyfile available
from timeit import timeit

from bokeh.io import output_file, show, reset_output, save, export_png
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh.plotting import figure
from bokeh.layouts import row, column, gridplot
from bokeh.io import output_file, show, reset_output, save, export_png
from bokeh.models.annotations import Title, Legend
from bokeh.models.widgets.markups import Div
from bokeh.models import LinearAxis, Range1d
from bokeh.palettes import Viridis7, Viridis4, Inferno7

class EpisodePlotterWrapper_multi_agent(gym.Wrapper):

    standard_output_props = [
        prp.altitude_sl_ft
        , prp.flight_path_deg
        , prp.roll_deg
        , prp.true_airspeed
        , prp.indicated_airspeed
    ]

    def __init__(self, env, output_props = []):
        super(EpisodePlotterWrapper_multi_agent, self).__init__(env)

        #append the restore data
        self.env_init_dicts.append({
            'output_props': output_props
        })
        self.env_classes.append(self.__class__.__name__)

        self.env = env
        self.step_time = self.env.dt  #use this to scale the x-axis

        #create a pandas dataframe to hold all episode data
        # |state | reward | done | action| per each step with property names as data
        self.action_variables = self.env.action_props
        state_variables = self.env.state_props
        reward_variables = [prp.Property('rwd_'+t.name, 'the reward obtained in this step by TaskAgent '+t.name) for t in self.env.task_list]
        done_variables = [prp.Property('done_'+t.name, 'the done flag for TaskAgent '+t.name) for t in self.env.task_list]
        task_agent_output_props = []
        [ task_agent_output_props.extend(t.get_props_to_output()) for t in self.env.task_list ]
        # self.state = np.empty(self.env.observation_space.shape)
        # self.reward = 0.0
        # self.done = False

        #stash away the properties for the individual panels for each AgentTask:
        self.task_names = [t.name for t in env.task_list]
        self.panel_contents = {}
        for i, t in enumerate(env.task_list):
            reward_component_names = [cmp.name for cmp in t.assessor.base_components]
            self.panel_contents[t.name] = {'panel1': {'setpoint_value_prop': t.setpoint_value_props[0],
                                                'action_prop': t.action_props[0],
                                                'current_value_prop': t.setpoint_props[0]},
                                           'panel2': {'reward_prop': reward_variables[i],
                                                'reward_component_names': reward_component_names},
                                           'panel3': {'obs_props': t.obs_props}
                                        }

        # add the output props coming as step() params in a given order to the recorder dataset 
        step_param_props = state_variables \
                         + reward_variables \
                         + done_variables \
                         + self.action_variables

        # prepare a set of props to be queried in the sim object by the episode plotter itself
        self_collected_output_props_set = set(task_agent_output_props) | set(output_props) | set(self.standard_output_props)
        self.self_collected_output_props = list(self_collected_output_props_set - set(step_param_props))
        # only add the self_collected props which are not yet in the step_param_props-list to the end of the recorder dataset
        recorder_data_set = step_param_props + self.self_collected_output_props
        self.recorderCols = list(map(lambda el:el.get_legal_name(),recorder_data_set))

        self.showNextPlotFlag = False
        self.exportNextPlotFlag = False
        self.save_to_csv = False
        self.firstRun = True #to determine if we're supposed to open a new Browser window

        self.recorderDictList = []   #see https://stackoverflow.com/a/17496530/2682209

        self.dirname = os.path.dirname(__file__) + '/../plots/{}'.format(datetime.datetime.now().strftime("%Y_%m_%d-%H:%M"))
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)
        
    def _collect_data(self):
        collected_data = [self.env.sim[prop] for prop in self.self_collected_output_props]
        return collected_data
    
    def step(self, actions_n):
        #let's move on to the next step
        self.newObs_n = self.env.step(actions_n)
        _, reward_n, done_n, info_n = self.newObs_n
        reward_components_dict_n = [info['reward_components'] for info in info_n]  #TODO: this is all very hacky. There must be a smoother way to handle the reward components
        reward_components_dict = {}
        [reward_components_dict.update(comp_dict) for comp_dict in reward_components_dict_n]
        state = np.array(self.env.state)
        #flatten the actions
        actions = []
        [actions.extend(act) for act in actions_n]

        data = np.concatenate( (state, reward_n, done_n, actions) ).tolist()
        dataDict = dict(zip(self.recorderCols, data + self._collect_data()))
        dataDict.update(reward_components_dict)
        self.recorderDictList.append(dataDict)
        if any(done_n) or self.env.is_terminal():
            if (self.showNextPlotFlag or self.exportNextPlotFlag or self.save_to_csv):
                dataRecorder = pd.DataFrame(self.recorderDictList)    
                if (self.save_to_csv):
                    #save the entire pandas frame to CSV file
                    csv_dir_name = os.path.join(self.dirname, '../csv')
                    if not os.path.exists(csv_dir_name):
                        os.mkdir(csv_dir_name)
                    filename = os.path.join(csv_dir_name, 'state_record_{}.csv'.format(datetime.datetime.now().strftime("%H:%M:%S")))
                    dataRecorder.to_csv(filename)
                # print(f"available properties for plotting:\n{dataRecorder.keys()}")   #this is handy if you want to change the plot to get the available data headings
                self.showGraph(dataRecorder)

        return self.newObs_n

    def close(self):
        #if env is closed before itself gets a done, show the graph is needed
        if (self.showNextPlotFlag or self.exportNextPlotFlag or self.save_to_csv):
            dataRecorder = pd.DataFrame(self.recorderDictList)    
            if (self.save_to_csv):
                #save the entire pandas frame to CSV file
                csv_dir_name = os.path.join(self.dirname, '../csv')
                if not os.path.exists(csv_dir_name):
                    os.mkdir(csv_dir_name)
                filename = os.path.join(csv_dir_name, 'state_record_{}.csv'.format(datetime.datetime.now().strftime("%H:%M:%S")))
                dataRecorder.to_csv(filename)
            # print(dataRecorder.keys())   #this is handy if you want to change the plot to get the available data headings
            self.showGraph(dataRecorder)


    def reset(self):
        self.recorderDictList = []   #see https://stackoverflow.com/a/17496530/2682209
        self.obs_n = self.env.reset()
        #save the initial state
        data = np.concatenate( (np.array(self.env.state), np.zeros(len(self.env.task_list)), np.zeros(len(self.env.task_list)), np.zeros(len(self.action_variables))) ).tolist()
        dataDict = dict(zip(self.recorderCols, data+ self._collect_data()))
        self.recorderDictList.append(dataDict)

        return self.obs_n
    
    def create_task_panels(self, data_frame):
        panels = {}
        top_left_x_range = None
        for i, t in enumerate(self.env.task_list):
            panels[t.name]={}
            # Panel 1: Setpoint and command panel
            pCtrl = figure(plot_width=800, plot_height=300)
            ctrl_legend = []
            # Setting the second y axis range name and range
            pCtrl.extra_y_ranges = {t.name+'_cmd': Range1d(start=-1, end=1)} # this should query the action space
            # Adding the second axis to the plot.  
            pCtrl.add_layout(LinearAxis(y_range_name=t.name+'_cmd', axis_label=t.name+'_cmd [norm.]'), 'right')
            name = self.panel_contents[t.name]['panel1']['action_prop'].get_legal_name() #maybe there are more entries in the future
            action_Line  = pCtrl.line(data_frame.index*self.step_time, data_frame[name], 
                                            line_width=1, y_range_name=t.name+'_cmd', color=Viridis4[1])
            ctrl_legend.append( (t.name+' Cmd.', [action_Line]) )                                            
            name = self.panel_contents[t.name]['panel1']['current_value_prop'].get_legal_name()
            current_value_line = pCtrl.line(data_frame.index*self.step_time, data_frame[name], 
                                            line_width=2, color=Viridis4[0])
            ctrl_legend.append( (name, [current_value_line]) )                                            
            name = self.panel_contents[t.name]['panel1']['setpoint_value_prop'].get_legal_name()
            setpoint_value_line = pCtrl.line(data_frame.index*self.step_time, data_frame[name], 
                                            line_width=2, color=Viridis4[3])
            ctrl_legend.append( (name, [setpoint_value_line]) )                                            
            
            ctrl_lg = Legend(items = ctrl_legend, location=(0, 10), glyph_width = 25, label_width = 190)
            ctrl_lg.click_policy="hide"
            pCtrl.add_layout(ctrl_lg, 'right')
            pCtrl.toolbar.active_scroll = pCtrl.toolbar.tools[1]    #this selects the WheelZoomTool instance                                          
            # Add the title...
            tCtrl = Title()
            tCtrl.text = 'Controlled Value over Time'
            pCtrl.title = tCtrl
            pCtrl.xaxis.axis_label = 'timestep [s]'
            pCtrl.yaxis[0].axis_label = 'Controlled Value'

            if not top_left_x_range:
                top_left_x_range = pCtrl.x_range
            else: 
                pCtrl.x_range = top_left_x_range
            panels[t.name].update({'panel1': pCtrl})

            #Panel 2: Rewards and reward components
            pRwd = figure(plot_width=1057, plot_height=300, x_range=top_left_x_range)
            rwd_cmp_lines = []
            reward_legend = []

            cmp_names = self.panel_contents[t.name]['panel2']['reward_component_names']
            for idx, rwd_component in enumerate(cmp_names):
                rwd_cmp_lines.append (
                    pRwd.line(data_frame.index*self.step_time, data_frame[rwd_component], line_width=2, color=Viridis7[idx%6])
                )
                reward_legend.append( (rwd_component, [rwd_cmp_lines[-1]]) )

            name = self.panel_contents[t.name]['panel2']['reward_prop'].get_legal_name()
            reward_line = pRwd.line(data_frame.index*self.step_time, data_frame[name], line_width=2, color=Viridis7[6])
            reward_legend.append( (name, [reward_line]) )

            reward_lg = Legend(items = reward_legend, location=(48, 10), glyph_width = 25, label_width = 190)
            reward_lg.click_policy="hide"
            pRwd.add_layout(reward_lg, 'right')
            pRwd.toolbar.active_scroll = pRwd.toolbar.tools[1]    #this selects the WheelZoomTool instance                                          
            #Add the title
            tReward = Title()
            tReward.text = f'{t.name}: actual Reward over {data_frame[name].size} Timesteps (∑ = {data_frame[name].sum():.2f})'
            pRwd.title = tReward
            pRwd.xaxis.axis_label = 'timestep [s]'
            pRwd.yaxis[0].axis_label = 'actual Reward [norm.]'
            panels[t.name].update({'panel2' : pRwd})


            #Panel 3: Presented Observations
            pState = figure(plot_width=1057, plot_height=300, x_range=top_left_x_range)
            # Setting the second y axis range name and range
            norm_state_extents = 10
            pState.extra_y_ranges = {"normalized_data": Range1d(start=-norm_state_extents, end=norm_state_extents )}
            # Adding the second axis to the plot.  
            pState.add_layout(LinearAxis(y_range_name="normalized_data", axis_label="normalized data"), 'right')
            state_lines = []
            state_legend = []
            normalized_state_lines = []
            obs_props_names = [o_prp.get_legal_name() for o_prp in self.panel_contents[t.name]['panel3']['obs_props']]
            for idx, state_name in enumerate(obs_props_names):
                if(data_frame[state_name].max() <= norm_state_extents and data_frame[state_name].min() >= -norm_state_extents):
                    normalized_state_lines.append(
                        pState.line(data_frame.index*self.step_time, data_frame[state_name], line_width=2, y_range_name="normalized_data", color=Inferno7[idx%7], visible=False)
                    )
                    state_legend.append( ("norm_"+state_name, [normalized_state_lines[-1]]) )                    
                else:     
                    state_lines.append(
                        pState.line(data_frame.index*self.step_time, data_frame[state_name], line_width=2, color=Inferno7[idx%7], visible=False)
                    )
                    state_legend.append( (state_name, [state_lines[-1]]) )                    
            pState.y_range.renderers = state_lines
            pState.extra_y_ranges.renderers = normalized_state_lines    #this does not quite work: https://stackoverflow.com/questions/48631530/bokeh-twin-axes-with-datarange1d-not-well-scaling

            lg_state = Legend(items = state_legend, location=(0, 10), glyph_width = 25, label_width = 190)
            lg_state.click_policy="hide"
            pState.add_layout(lg_state, 'right')
            pState.toolbar.active_scroll = pState.toolbar.tools[1]    #this selects the WheelZoomTool instance                                          
            #Add the title
            tState = Title()
            tState.text = 'Actual Observation presented to Agent'
            pState.title = tState
            pState.xaxis.axis_label = 'timestep [s]'
            pState.yaxis[0].axis_label = 'state data'

            panels[t.name].update({'panel3': pState})
        
        return panels


    def showGraph(self,data_frame):

        #create the plot panels for the Agent_Tasks
        panels = self.create_task_panels(data_frame)

        top_left_x_range = panels[list(panels.keys())[0]]['panel1'].x_range

        #Altitude over ground
        pAltitude = figure(plot_width=800, plot_height=300, x_range=top_left_x_range)
        alti_legend = []
        # Setting the second y axis range name and range
        pAltitude.extra_y_ranges = {"speed": Range1d(50, 120)}
        # Adding the second axis to the plot.  
        pAltitude.add_layout(LinearAxis(y_range_name="speed", axis_label="IAS, TAS [Knots]"), 'right')

        altitudeLine = pAltitude.line(data_frame.index*self.step_time, data_frame['position_h_sl_ft'], line_width=2, color=Viridis4[2])
        alti_legend.append( ("Altitude [ftsl]", [altitudeLine]) ) 
        kiasLine = pAltitude.line(data_frame.index*self.step_time, data_frame['velocities_vc_kts'], line_width=2, y_range_name="speed", color=Viridis4[1])
        alti_legend.append( ("Indicated Airspeed [KIAS]", [kiasLine]) ) 
        tasLine = pAltitude.line(data_frame.index*self.step_time, data_frame['velocities_vtrue_kts'], line_width=2, y_range_name="speed", color=Viridis4[0])
        alti_legend.append( ("True Airspeed [KAS]", [tasLine]) ) 
        pAltitude.extra_y_ranges.renderers = [kiasLine, tasLine]    #this does not quite work: https://stackoverflow.com/questions/48631530/bokeh-twin-axes-with-datarange1d-not-well-scaling
        pAltitude.y_range.renderers = [altitudeLine]

        lg_alti = Legend(items = alti_legend, location=(0, 10), glyph_width = 25, label_width = 190)
        lg_alti.click_policy="hide"
        pAltitude.add_layout(lg_alti, 'right')

        tAlti = Title()
        tAlti.text = 'Altitude and Speed [IAS, TAS] over Timesteps'
        pAltitude.title = tAlti
        pAltitude.xaxis.axis_label = 'timestep [s]'
        pAltitude.yaxis[0].axis_label = 'Altitude [ftsl]'
        pAltitude.legend.location="center_right"

        pSideslip = figure(plot_width=800, plot_height=300, x_range=top_left_x_range)
        slip_legend = []

        slip_skid_line = pSideslip.line(data_frame.index*self.step_time, data_frame['aero_beta_deg'], line_width=2, color=Viridis4[2])
        slip_legend.append( ("Sideslip", [slip_skid_line]) ) 
        pSideslip.y_range.renderers = [slip_skid_line]

        lg_slip = Legend(items = slip_legend, location=(0, 10), glyph_width = 25, label_width = 190)
        lg_slip.click_policy="hide"
        pSideslip.add_layout(lg_slip, 'right')

        tSlip = Title()
        tSlip.text = 'Sideslip'
        pSideslip.title = tSlip
        pSideslip.xaxis.axis_label = 'timestep [s]'
        pSideslip.yaxis[0].axis_label = 'Sideslip [deg]'
        pSideslip.legend.location="center_right"


        #activate the zooming on all plots
        #this is not nice, but this not either: https://stackoverflow.com/questions/49282688/how-do-i-set-default-active-tools-for-a-bokeh-gridplot
        pAltitude.toolbar.active_scroll = pAltitude.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        pSideslip.toolbar.active_scroll = pSideslip.toolbar.tools[1]    #this selects the WheelZoomTool instance 

        reset_output()

        # if self.env.meta_dict['model_type'] == 'trained':
        #     discriminator = self.env.meta_dict['model_base_name']+"_%+.2f" % (data_frame['reward'].sum())
        #     self.env.meta_dict['model_discriminator'] = discriminator
        # else: 
        #     discriminator = self.env.meta_dict['model_discriminator']

        panel_grid_t = [ [panels[name]['panel1'],panels[name]['panel2'],panels[name]['panel3']] for name in self.task_names]
        panel_grid= list(zip(*panel_grid_t))

        # add the additional plots
        panel_grid.append([pAltitude, pSideslip])
        
        panel_grid_plot = gridplot(panel_grid, toolbar_location='right', sizing_mode='stretch_width')

        #for string formatting look here: https://pyformat.info/

        titleString = "Run Date: {}; ".format(datetime.datetime.now().strftime("%c"))
        
        if 'train_step' in self.env.meta_dict:
            titleString += "Training Step: {}; ".format(self.env.meta_dict['train_step'])
        if 'episode_number' in self.env.meta_dict:
            titleString += "Episode: {}; ".format(self.env.meta_dict['episode_number'])
        # titleString += "Total Reward: {:.2f}; ".format(data_frame['reward'].sum())
        # titleString += "Model Discriminator: {};".format(self.env.meta_dict['model_discriminator'])
        header_col = column(
            Div(text="<h1>" + self.env.unwrapped.spec.id + 
            (" - " + self.env.meta_dict['env_info']) if 'env_info' in self.meta_dict else "" + 
            "</h1>"), 
            Div(text="<h2>"+titleString+"</h2>")) 

        grid = gridplot([[header_col],[panel_grid_plot]], toolbar_location=None, sizing_mode='stretch_width')

        html_output_name = os.path.join(self.dirname, 'glideAngle_Elevator_latest.html')
        if self.showNextPlotFlag:
            output_file(html_output_name, mode='absolute') #use mode='absolute' to make it work offline with the js and css installed in the bokeh package locally
            if self.firstRun:
                show(grid)  #opens up a new browser window
                self.firstRun = False
            else:
                save(grid)  #just updates the HTML; Manual F5 in browser required :-(, (There must be a way to push...)

        
        # if self.exportNextPlotFlag:
        #     base_filename = os.path.join(self.dirname, 'glideAngle_Elevator_Reward_{:.2f}_time_{}'.format(data_frame['reward'].sum(), datetime.datetime.now().strftime("%H:%M:%S")))
        #     # @timeit   TODO: die sourcen werden nicht gefunden
        #     if self.showNextPlotFlag:
        #         #we keep the html as well for easy exploration
        #         shutil.copyfile(html_output_name, base_filename+'.html')
        #     def export(webpage):
        #         filename = base_filename + '.png'
        #         export_png(webpage, filename= filename)
        #     export(webpage)

        self.showNextPlotFlag = False   #only show the plot once and then reset
        self.exportNextPlotFlag = False
        print("Output Plot generated: "+titleString)

    def showNextPlot(self, show = False, export = False, save_to_csv = False):
        self.showNextPlotFlag = show
        self.exportNextPlotFlag = export
        self.save_to_csv = save_to_csv
    
    def prepare_plot_meta(self):
        env_id = "<h1>" + self.env.unwrapped.spec.id + "</h1>"

        meta_info_string = ""
        for item in self.env.meta_dict.items():
            meta_info_string += f"{item}<br>"
        return env_id + "<h2>"+"</h2>" + meta_info_string
    
    def get_meta_info_table(self):
        data = {'keys': list(self.env.meta_dict.keys()),
                'vals': list(self.env.meta_dict.values())}
        source = ColumnDataSource(data = data)

        columns = [
                TableColumn(field="keys", title="Key"),
                TableColumn(field="vals", title="Value"),
            ]
        return DataTable(source=source, columns=columns, fit_columns= True, reorderable=False, editable=True)

        
        

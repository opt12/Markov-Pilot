import gym
import gym.spaces
import markov_pilot.environment.properties as prp
import numpy as np
import pandas as pd
import datetime
import os
import math
import shutil   #to have copyfile available
from timeit import timeit
import time

from bokeh.io import output_file, show, reset_output, save, export_png
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh.plotting import figure
from bokeh.layouts import row, column, gridplot
from bokeh.io import output_file, show, reset_output, save, export_png
from bokeh.models.annotations import Title, Legend
# from bokeh.models.widgets.markups import Div
from bokeh.models import Button, Div, CustomJS, CheckboxGroup
from bokeh import events
from bokeh.models import LinearAxis, Range1d
from bokeh.palettes import Viridis7, Viridis4, Inferno7

class EpisodePlotterWrapper_multi(gym.Wrapper):

    standard_output_props = [
        prp.altitude_sl_ft
        , prp.flight_path_deg
        , prp.roll_deg
        , prp.true_airspeed
        , prp.indicated_airspeed
    ]

    def __init__(self, env, output_props = []):
        super(EpisodePlotterWrapper_multi, self).__init__(env)

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
        self.reward_variables = [prp.Property('rwd_'+t.name, 'the reward obtained in this step by Task '+t.name) for t in self.env.task_list]
        done_variables = [prp.Property('done_'+t.name, 'the done flag for Task '+t.name) for t in self.env.task_list]
        task_agent_output_props = []
        [ task_agent_output_props.extend(t.get_props_to_output()) for t in self.env.task_list ]
        # self.state = np.empty(self.env.observation_space.shape)
        # self.reward = 0.0
        # self.done = False

        #stash away the properties for the individual panels for each FlightTask:
        #also stash away the setpoint_names, the error_names and the delta_cmd_names for the evaluation
        self.task_names = [t.name for t in env.task_list]
        self.panel_contents = {}
        # self.setpoint_names = ['setpoint_attitude_phi_deg', 'setpoint_flight_path_gamma_deg']
        # self.error_names = ['error_aileron_err', 'error_elevator_err']
        # self.delta_cmd_names = ['info_aileron_delta_cmd', 'info_elevator_delta_cmd']

        for i, t in enumerate(env.task_list):
            reward_component_names = [cmp.name for cmp in t.assessor.base_components]
            self.panel_contents[t.name] = {'panel1': {},
                                           'panel2': {'reward_prop': self.reward_variables[i],
                                                'reward_component_names': reward_component_names},
                                           'panel3': {'obs_props': t.obs_props}
                                        }
            if t.action_props:
                self.panel_contents[t.name]['panel1'].update({'action_prop': t.action_props[0]})

            if t.setpoint_props:
                self.panel_contents[t.name]['panel1'].update({'setpoint_value_prop': t.setpoint_value_props[0],
                                                'current_value_prop': t.setpoint_props[0]})

        # add the output props coming as step() params in a given order to the recorder dataset 
        step_param_props = state_variables \
                         + self.reward_variables \
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

       
    def _collect_data(self):
        collected_data = [self.env.sim[prop] for prop in self.self_collected_output_props]
        return collected_data
    
    def step(self, actions_n):
        # pylint: disable=method-hidden
        #let's move on to the next step
        self.newObs_n = self.env.step(actions_n)
        _, reward_n, done_n, info_n = self.newObs_n
        reward_components_dict_n = [info['reward_components'] for info in info_n]  #TODO: this is all very hacky. There must be a smoother way to handle the reward components
        reward_components_dict = {}
        [reward_components_dict.update(comp_dict) for comp_dict in reward_components_dict_n]
        state = np.array(self.env.state)
        #flatten the actions
        actions = []
        [actions.extend(act) for act in actions_n if act != None]

        data = np.concatenate( (state, reward_n, done_n, actions) ).tolist()
        dataDict = dict(zip(self.recorderCols, data + self._collect_data()))
        dataDict.update(reward_components_dict)
        self.recorderDictList.append(dataDict)
        if any(done_n) or self.env.is_terminal():
            if (self.showNextPlotFlag or self.exportNextPlotFlag or self.save_to_csv):
                dataRecorder = pd.DataFrame(self.recorderDictList)    
                if self.save_to_csv and self.save_path:
                    #save the entire pandas frame to CSV file
                    csv_dir_name = os.path.join(self.save_path, 'csv')
                    #create the directory for this csv
                    os.makedirs(csv_dir_name, exist_ok=True)
                    filename = os.path.join(csv_dir_name, 'state_record_{}.csv'.format(datetime.datetime.now().strftime("%H-%M-%S")))
                    dataRecorder.to_csv(filename)
                # print(f"available properties for plotting:\n{dataRecorder.keys()}")   #this is handy if you want to change the plot to get the available data headings
                self._show_graph(dataRecorder)

        return self.newObs_n

    def close(self):
        #if env is closed before itself gets a done, show the graph is needed
        if (self.showNextPlotFlag or self.exportNextPlotFlag or self.save_to_csv):
            dataRecorder = pd.DataFrame(self.recorderDictList)    
            if self.save_to_csv and self.save_path:
                #save the entire pandas frame to CSV file
                csv_dir_name = os.path.join(self.save_path, 'csv')
                #create the directory for this csv
                os.makedirs(csv_dir_name, exist_ok=True)
                filename = os.path.join(csv_dir_name, 'state_record_{}.csv'.format(datetime.datetime.now().strftime("%H-%M-%S")))
                dataRecorder.to_csv(filename)
            # print(dataRecorder.keys())   #this is handy if you want to change the plot to get the available data headings
            self._show_graph(dataRecorder)

    def reset(self):
        # pylint: disable=method-hidden
        self.recorderDictList = []   #see https://stackoverflow.com/a/17496530/2682209
        self.obs_n = self.env.reset()
        #save the initial state
        data = np.concatenate( (np.array(self.env.state), np.zeros(len(self.env.task_list)), np.zeros(len(self.env.task_list)), np.zeros(len(self.action_variables))) ).tolist()
        dataDict = dict(zip(self.recorderCols, data+ self._collect_data()))
        self.recorderDictList.append(dataDict)

        return self.obs_n
    
    def _create_task_panels(self, data_frame):
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
            try:
                name = self.panel_contents[t.name]['panel1']['action_prop'].get_legal_name() #maybe there are more entries in the future
                action_Line  = pCtrl.line(data_frame.index*self.step_time, data_frame[name], 
                                            line_width=1, y_range_name=t.name+'_cmd', color=Viridis4[1])
                ctrl_legend.append( (t.name+' Cmd.', [action_Line]) )                                            
            except KeyError:
                #we have no action prop, only setpoints to be evaluated
                pass
            try:
                name = self.panel_contents[t.name]['panel1']['current_value_prop'].get_legal_name()
                current_value_line = pCtrl.line(data_frame.index*self.step_time, data_frame[name], 
                                            line_width=2, color=Viridis4[0])
                ctrl_legend.append( (name, [current_value_line]) )                                            
                name = self.panel_contents[t.name]['panel1']['setpoint_value_prop'].get_legal_name()
                setpoint_value_line = pCtrl.line(data_frame.index*self.step_time, data_frame[name], 
                                            line_width=2, color=Viridis4[3])
                ctrl_legend.append( (name, [setpoint_value_line]) )
            except KeyError:
                #there is no setpoint to be displayed, only actuations
                pass
            
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
            tReward.text = f'{t.name}: last Reward over {data_frame[name].size-1} Timesteps (∑ = {data_frame[name].sum():.2f})'
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

    def _show_graph(self,data_frame):

        #create the plot panels for the Agent_Tasks
        panels = self._create_task_panels(data_frame)

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

        ts = time.time()
        overshoot_frames_per_task = self._analyze_overshoot(data_frame)
        overshoot_divs = [Div(text = ovs_fr.round(3).to_html(), width = 600) for ovs_fr in overshoot_frames_per_task]
        print("Overshoot analysis done in %.2f sec" % (time.time() - ts))

        ts = time.time()
        settlement_times_per_task = self._analyze_settle_times(data_frame)
        settlement_divs = [Div(text = settle_fr.round(3).to_html(), width = 600) for settle_fr in settlement_times_per_task]
        print("Settlement analysis done in %.2f sec" % (time.time() - ts))

        panel_grid = []
        panel_grid.append([Div(text='<h3>'+t.name+'</h3>', id='div_'+t.name) for t in self.env.task_list])

        # to switch on and off the statistics panels, this is unfortuntely the best, I could achive
        # https://stackoverflow.com/a/52416676/2682209
        cols = []
        checkbox = CheckboxGroup(labels=["show stats"], active=[], width=100)   #checkbox is added to header_col later on

        for i, t in enumerate(self.env.task_list):
            # overshoot_stat = overshoot_divs[i]
            c = column(Div())   #empty for the beginning
            cols.append(c)

        callback = CustomJS(args=dict(overshoot_divs=overshoot_divs, settlement_divs=settlement_divs, cols=cols, checkbox=checkbox), code="""
                    for (var j = 0; j < cols.length; j++) {
                        console.log('col', j)
                        const children = []
                        for (const i of checkbox.active) {
                            console.log('active', i)
                            children.push(overshoot_divs[j])
                            children.push(settlement_divs[j])
                        } 
                        console.log('children', children)
                        cols[j].children = children
                    }
                    """)
        checkbox.js_on_change('active', callback)



        # show_stats_btn = [Div(text="""
        # <button onclick="display_event(%s)">Try it</button>
        # """ %t.name for t in self.env.task_list]
        # for t, b in zip(self.env.task_list, show_stats_btn):
        #     b.tags = ['id', 'btn_'+t.name]
        # panel_grid.append(show_stats_btn)
        # [b.js_on_event(events.ButtonClick, display_event(b, t.name)) for t, b in zip(self.env.task_list, show_stats_btn)]

        # panel_grid.append(chkbxs)
        panel_grid.append(cols)

        panel_grid_t = [ [panels[name]['panel1'],panels[name]['panel2'],panels[name]['panel3']] for name in self.task_names]
        [panel_grid.append(fig) for fig in list(zip(*panel_grid_t))]

        # add the additional plots
        panel_grid.append([pAltitude, pSideslip])
        
        panel_grid_plot = gridplot(panel_grid, toolbar_location='right', sizing_mode='stretch_width')

        #for string formatting look here: https://pyformat.info/
        titleString = ''
        if 'experiment_name' in self.env.meta_dict:
            titleString += "{}: ".format(self.env.meta_dict['experiment_name'])
        titleString += "Run Date: {}; ".format(datetime.datetime.now().strftime("%c"))
        
        if 'train_step' in self.env.meta_dict:
            titleString += "Training Step: {}; ".format(self.env.meta_dict['train_step'])
        if 'episode_number' in self.env.meta_dict:
            titleString += "Episode: {}; ".format(self.env.meta_dict['episode_number'])
        if 'csv_line_nr' in self.env.meta_dict:
            titleString += "Env in CSV line: {}; ".format(self.env.meta_dict['csv_line_nr'])
        # titleString += "Total Reward: {:.2f}; ".format(data_frame['reward'].sum())
        # titleString += "Model Discriminator: {};".format(self.env.meta_dict['model_discriminator'])
        header_col = column(
            Div(text="<h1>" + self.env.unwrapped.spec.id + 
            (" - " + self.env.meta_dict['env_info']) if 'env_info' in self.meta_dict else "" + 
            "</h1>"), 
            row(
            Div(text="<h2>"+titleString+"</h2>", width=1200), checkbox) )

        webpage = gridplot([[header_col],[panel_grid_plot]], toolbar_location=None, sizing_mode='stretch_width')

        base_filename = 'Run_'+ '_'.join([tsk.name for tsk in self.task_list])
        html_output_name = os.path.join(self.save_path,'plots', base_filename+'_latest.html')
        os.makedirs(os.path.dirname(html_output_name), exist_ok=True)

        if self.showNextPlotFlag:
            output_file(html_output_name, mode='inline') #mode='absolute') #use mode='absolute' to make it work offline with the js and css installed in the bokeh package locally
            if self.firstRun:
                show(webpage)  #opens up a new browser window
                self.firstRun = False
            else:
                save(webpage)  #just updates the HTML; Manual F5 in browser required :-(, (There must be a way to push...)

        
        if self.exportNextPlotFlag and self.save_path:
            #build the filename including the individual rewards
            task_rewards = [self.reward_variables[i].get_legal_name() for i in range(len(self.task_list))]
            task_names_with_rewards = [t.name+'_'+f'{data_frame[task_rewards[i]].sum():.2f}' for i, t in enumerate(self.task_list)]
            name_with_rewards = 'Run_' + '_'.join(task_names_with_rewards)+'time_{}'.format(datetime.datetime.now().strftime("%H-%M-%S"))
            base_filename = os.path.join(self.save_path, 'plots', name_with_rewards) # 'glideAngle_Elevator_Reward_{:.2f}_time_{}'.format(data_frame['reward'].sum(), datetime.datetime.now().strftime("%H-%M-%S")))
            if self.showNextPlotFlag:
                #we keep the html as well for easy exploration
                shutil.copyfile(html_output_name, base_filename+'.html')
            def export(webpage):
                png_filename = base_filename + '.png'
                webpage.width = 1800    #set the width of the page instead of passing a width parameter to the export; https://stackoverflow.com/a/61563173/2682209
                export_png(webpage, filename= png_filename) #TODO: the width parameter is ignored in bokeh/io/export.py get_layout_html() as webpage isn't a Plot
            export(gridplot([[header_col],[panel_grid_plot]], toolbar_location=None))

        self.showNextPlotFlag = False   #only show the plot once and then reset
        self.exportNextPlotFlag = False
        print("Output Plot generated: "+titleString)

    def showNextPlot(self, show = False, export = False, save_to_csv = False):
        self.showNextPlotFlag = show
        self.exportNextPlotFlag = export
        self.save_to_csv = save_to_csv
    
    def _prepare_plot_meta(self):
        env_id = "<h1>" + self.env.unwrapped.spec.id + "</h1>"

        meta_info_string = ""
        for item in self.env.meta_dict.items():
            meta_info_string += f"{item}<br>"
        return env_id + "<h2>"+"</h2>" + meta_info_string
    
    def _get_meta_info_table(self):
        data = {'keys': list(self.env.meta_dict.keys()),
                'vals': list(self.env.meta_dict.values())}
        source = ColumnDataSource(data = data)

        columns = [
                TableColumn(field="keys", title="Key"),
                TableColumn(field="vals", title="Value"),
            ]
        return DataTable(source=source, columns=columns, fit_columns= True, reorderable=False, editable=True)

    def _prepare_analysis_data(self, data_frame):
        setpoint_names = []
        error_names = []
        delta_cmd_names = []
        intial_setpoints = []
        for t in self.env.task_list:
            try:
                setpoint_name = t.setpoint_value_props[0].get_legal_name()
                intial_setpoint = data_frame[t.setpoint_props[0].get_legal_name()][0]    #the initial setpoin is the first entry of the current value in the data_frameS
            except IndexError:
                #there is no setpoint in that task
                setpoint_name = ''  #set it to empty as a marker
                intial_setpoint = float('nan') 
            try:
                error_name = t.prop_error.get_legal_name()
            except AttributeError:
                #there is no error in that task (as there is no setpoint)
                error_name = ''   #set it to empty as a marker
            try:
                delta_cmd_name = t.prop_delta_cmd.get_legal_name()
            except AttributeError:
                #there is no command in that task
                delta_cmd_name = ''   #set it to empty as a marker
            
            setpoint_names.append(setpoint_name)
            intial_setpoints.append(intial_setpoint)
            error_names.append(error_name)
            delta_cmd_names.append(delta_cmd_name)

        #check when the setpoint changes for each task individually
        change_idxs = [np.array(data_frame[setp_name].diff()[data_frame[setp_name].diff() != 0].index.values) 
                            for setp_name in setpoint_names if setp_name!='']
        #an event occurs whenever a setpoint changes in any task
        event_idxs = np.unique(np.sort(np.hstack(change_idxs)))
        #the setpoints at each event
        setpoints = []
        setpoint_changes = []
        for channel,setp_name in enumerate(setpoint_names):
            if setp_name!='':
                setpt = np.array(data_frame[setpoint_names[channel]])[event_idxs]
                sp_ch = np.array(data_frame[setpoint_names[channel]].diff()[event_idxs])
                if math.isnan(sp_ch[0]):
                    sp_ch[0] = data_frame[setpoint_names[channel]][change_idxs[0][0]] -intial_setpoints[channel]
            else: 
                setpt = np.empty(len(event_idxs))
                setpt[:] = np.NaN
                sp_ch = np.empty(len(event_idxs))
                sp_ch[:] = np.NaN
            setpoints.append(setpt)
            setpoint_changes.append(sp_ch)
        event_idxs = np.append(event_idxs, data_frame.last_valid_index()+1)

        return setpoint_names, error_names, delta_cmd_names, event_idxs, setpoints, setpoint_changes

    def _analyze_overshoot(self, data_frame):
        #see http://localhost:8888/notebooks/git/gym-jsbsim-eee/testruns/generic/csv/evaluate_control.ipynb#

        setpoint_names, error_names, delta_cmd_names, event_idxs, setpoints, setpoint_changes = self._prepare_analysis_data(data_frame)

        dt = self.step_time
        min_length = 10
        epsilons = [0.5, 0.1, 0.05, 0.01]

        overshoot_frames_per_task = []
        for channel, t in enumerate(self.env.task_list):
            if error_names[channel] != '':
                #split the dataframe in event-wise segments
                data_segs = [data_frame.loc[event_idxs[i]:event_idxs[i+1]-1] for i in range(len(event_idxs)-1)]
                setpoint_current = setpoints[channel]
                setpoint_changes_current = setpoint_changes[channel]
                #calculate all overshoot values
                max_segs = [data_segs[i][error_names[channel]].max() for i in range(len(data_segs))]
                max_idxs = [data_segs[i][error_names[channel]].idxmax() for i in range(len(data_segs))]
                min_segs = [data_segs[i][error_names[channel]].min() for i in range(len(data_segs))]
                min_idxs = [data_segs[i][error_names[channel]].idxmin() for i in range(len(data_segs))]

                #the overshoot and its idx depending on the change direction
                overshoot = []
                overshoot_idxs = []
                for i in range(len(data_segs)):
                    if setpoint_changes[channel][i] > 0:
                        ovs = max_segs[i] 
                        ovs_idx = int(max_idxs[i])
                    elif setpoint_changes[channel][i] < 0:
                        ovs = min_segs[i] 
                        ovs_idx = int(min_idxs[i])
                    else:
                        #the change is 0, so we use the max absolute value
                        if abs(max_segs[i]) > abs(min_segs[i]):
                            ovs = max_segs[i] 
                            ovs_idx = int(max_idxs[i])
                        else:
                            ovs = min_segs[i] 
                            ovs_idx = int(min_idxs[i])
                    overshoot.append(ovs)
                    overshoot_idxs.append(ovs_idx)
                overshoot_idxs = np.array(overshoot_idxs)  #otherwise, we canot multiply it with dt afterwards

                # overshoot = [max_segs[i] if setpoint_changes[channel][i] > 0 else min_segs[i] 
                #             for i in range(len(data_segs))]
                # #th eindex when the peak occurs
                # overshoot_idxs = np.array([int(max_idxs[i]) if setpoint_changes[channel][i] > 0 else min_idxs[i] 
                #             for i in range(len(data_segs))])
                #the overshoot relative to the change
                overshoot_relative = [ovs / setpoint_changes[channel][i] if setpoint_changes[channel][i] != 0 else float('nan')
                                    for i, ovs in enumerate(overshoot)]
                #the min/max values of the peaks
                extreme_values = [data_frame[error_names[channel]][ovs_idx]+ data_frame[setpoint_names[channel]][ovs_idx]
                                for ovs_idx in overshoot_idxs]

                #the absolute and squared errors
                abs_errors = [data_frame[error_names[channel]][start:end].abs() for start, end in (zip(event_idxs[:-1], event_idxs[1:]-1))]
                squared_errors = [data_frame[error_names[channel]][start:end]**2 for start, end in (zip(event_idxs[:-1], event_idxs[1:]-1))]

                #the mean values over each event-segment
                abs_mean = [abse.mean() for abse in abs_errors]
                msqe = [sqe.mean() for sqe in squared_errors]
            else:
                overshoot_idxs = np.empty(len(event_idxs[:-1]))
                setpoint_current = np.empty(len(event_idxs[:-1]))
                extreme_values = np.empty(len(event_idxs[:-1]))
                setpoint_changes_current = np.empty(len(event_idxs[:-1]))
                overshoot = np.empty(len(event_idxs[:-1]))
                overshoot_relative = np.empty(len(event_idxs[:-1]))
                abs_mean = np.empty(len(event_idxs[:-1]))
                msqe = np.empty(len(event_idxs[:-1]))
                overshoot_idxs[:] = np.NaN
                setpoint_current[:] = np.NaN
                extreme_values[:] = np.NaN
                setpoint_changes_current[:] = np.NaN
                overshoot[:] = np.NaN
                overshoot_relative[:] = np.NaN
                abs_mean[:] = np.NaN
                msqe[:] = np.NaN

            if delta_cmd_names[channel] != '':
                #calculate the actuation energies
                deltas = [data_frame[delta_cmd_names[channel]][start:end]**2 for start, end in (zip(event_idxs[:-1], event_idxs[1:]-1))]
                actuation_energies = [delta.sum() for delta in deltas]
            else:
                actuation_energies = np.empty(len(event_idxs[:-1]))
                actuation_energies[:] = np.NaN

            #put everything into a new dataframe 
            overshoot_frame = pd.DataFrame(data=[
                                                event_idxs[:-1]*dt, 
                                                overshoot_idxs*dt, 
                                                (overshoot_idxs-event_idxs[:-1])*dt, 
                                                setpoint_current, 
                                                extreme_values, 
                                                setpoint_changes_current, 
                                                overshoot, 
                                                overshoot_relative,
                                                abs_mean,
                                                msqe,
                                                actuation_energies],
                                            index=[
                                                'event_time',
                                                'peak_time',
                                                'delay_secs', 
                                                'setpoint', 
                                                'actual_value', 
                                                'setpoint_change', 
                                                'abs_overshoot', 
                                                'rel_overshoot',
                                                'abs_mean',
                                                'MSE',
                                                'actuation_energy'])
            
            overshoot_frames_per_task.append(overshoot_frame)
        return overshoot_frames_per_task

    def _analyze_settle_times(self, data_frame):
        #see http://localhost:8888/notebooks/git/gym-jsbsim-eee/testruns/generic/csv/evaluate_control.ipynb#

        setpoint_names, error_names, delta_cmd_names, event_idxs, setpoints, setpoint_changes = self._prepare_analysis_data(data_frame)

        dt = self.step_time
        min_length = 10
        epsilons = [0.5, 0.1, 0.05, 0.01]

        settlement_times_per_task = []
        for channel, t in enumerate(self.env.task_list):
            settle_times = []
            for eps in epsilons:
                if error_names[channel] != '':
                    # print(f'*** Settlement to ±{eps}:')
                    settle_times_per_eps = []
                    for evt in range(len(event_idxs)-1):
                        #check the last min_length steps
                        start = event_idxs[evt+1]
                            #the last min_length steps are witin bounds, so let's go forward
                        for start in range (event_idxs[evt+1]-1, event_idxs[evt]-1, -1):
                            if abs(data_frame[error_names[channel]][start]) > eps:
                                start = start +1
                                break
                        if not all(data_frame[error_names[channel]][start:event_idxs[evt+1]].abs().lt(eps)) \
                            or event_idxs[evt+1]-start < min_length:
                            last = event_idxs[evt+1]
                            did_settle = False
                        else:
                            last = event_idxs[evt+1]-1
                            did_settle = True
                        settle_times_per_eps.append((start-event_idxs[evt])*dt if did_settle else float('nan'))  
                else:
                    settle_times_per_eps = np.empty(len(event_idxs[:-1]))
                    settle_times_per_eps[:] = np.NaN
                settle_times.append(settle_times_per_eps)
            #put everything into a new dataframe 
            data_for_frame = [setpoints[channel]]
            data_for_frame.append(setpoint_changes[channel])
            [data_for_frame.append(row) for row in settle_times]

            settle_times_frame = pd.DataFrame(data_for_frame, index=['setpoints', 'setpoint_changes', *epsilons])

            settlement_times_per_task.append(settle_times_frame)
        return settlement_times_per_task






        

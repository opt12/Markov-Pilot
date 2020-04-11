import gym
import gym.spaces
import gym_jsbsim.properties as prp
import numpy as np
import pandas as pd
import datetime
import os
import shutil   #to have copyfile available
from bokeh.io import output_file, show, reset_output, save, export_png
from bokeh.models import ColumnDataSource, DataTable, TableColumn
import math

from timeit import timeit

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
        reward_components_dict_n = [info['reward_components'] for info in info_n]  #TODO: this is all very hacky. Ther must be a smoother way to handle the reward components
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
                print(f"available properties for plotting:\n{dataRecorder.keys()}")   #this is handy if you want to change the plot to get the available data headings
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
    
    def showGraph(self,data_frame):
        from bokeh.plotting import figure
        from bokeh.layouts import row, column, gridplot
        from bokeh.io import output_file, show, reset_output, save, export_png
        from bokeh.models.annotations import Title, Legend
        from bokeh.models.widgets.markups import Div
        from bokeh.models import LinearAxis, Range1d
        from bokeh.palettes import Viridis4, Inferno7

        # GlideAngle and Elevator
        pElev = figure(plot_width=800, plot_height=400)
        # Setting the second y axis range name and range
        pElev.extra_y_ranges = {"elevator": Range1d(start=-1, end=1)}
        # Adding the second axis to the plot.  
        pElev.add_layout(LinearAxis(y_range_name="elevator", axis_label="Elevator Cmd [norm.]"), 'right')
        elevatorLine  = pElev.line(data_frame.index*self.step_time, data_frame['fcs_elevator_cmd_norm'], line_width=1, y_range_name="elevator", color=Viridis4[1], legend_label = "Elevator Cmd.")
        errorIntElevLine = pElev.line(data_frame.index*self.step_time, data_frame['error_elevator_int'], line_width=1, color=Viridis4[2], legend_label = "Error Integral.")
        gammaLine = pElev.line(data_frame.index*self.step_time, data_frame['flight_path_gamma_deg'], line_width=2, color=Viridis4[0], legend_label="Path angle")
        targetGammaLine = pElev.line(data_frame.index*self.step_time, data_frame['setpoint_elevator_setpoint'], line_width=2, color=Viridis4[3], legend_label="Target Path angle")
        # aoaLine = pElev.line(data_frame.index*self.step_time, data_frame['aero_alpha_deg'], line_width=1, color=Viridis4[2], legend_label="AoA", visible = False)

        # RollAngle and Aileron
        pAileron = figure(plot_width=800, plot_height=400, x_range=pElev.x_range)
        # Setting the second y axis range name and range
        pAileron.extra_y_ranges = {"aileron": Range1d(start=-1, end=1)}
        # Adding the second axis to the plot.  
        pAileron.add_layout(LinearAxis(y_range_name="aileron", axis_label="Aileron Cmd [norm.]"), 'right')

        aileronLine  = pAileron.line(data_frame.index*self.step_time, data_frame['fcs_aileron_cmd_norm'], line_width=1, y_range_name="aileron", color=Viridis4[1], legend_label = "Aileron Cmd.")
        # errorAilLine = pAileron.line(data_frame.index*self.step_time, data_frame['error_aileron_err'],    line_width=1, color=Viridis4[2], legend_label = "Roll angle dev.")
        # deltaAileronLine = pAileron.line(data_frame.index*self.step_time, data_frame['info_delta_cmd_aileron'], line_width=1, y_range_name="aileron", color=Viridis4[2], legend_label = "Δ Ail. Cmd.")
        phiLine = pAileron.line(data_frame.index*self.step_time, data_frame['attitude_phi_deg'], line_width=2, color=Viridis4[0], legend_label="Roll angle")
        targetPhiLine = pAileron.line(data_frame.index*self.step_time, data_frame['setpoint_aileron_setpoint'], line_width=2, color=Viridis4[3], legend_label="Target Roll angle")
        

        #Altitude over ground
        pAltitude = figure(plot_width=800, plot_height=300, x_range=pElev.x_range)
        # Setting the second y axis range name and range
        pAltitude.extra_y_ranges = {"speed": Range1d(50, 120)}
        # Adding the second axis to the plot.  
        pAltitude.add_layout(LinearAxis(y_range_name="speed", axis_label="IAS, TAS [Knots]"), 'right')

        altitudeLine = pAltitude.line(data_frame.index*self.step_time, data_frame['position_h_sl_ft'], line_width=2, color=Viridis4[2], legend_label = "Altitude [ftsl]")
        kiasLine = pAltitude.line(data_frame.index*self.step_time, data_frame['velocities_vc_kts'], line_width=2, y_range_name="speed", color=Viridis4[1], legend_label = "Indicated Airspeed [KIAS]")
        tasLine = pAltitude.line(data_frame.index*self.step_time, data_frame['velocities_vtrue_kts'], line_width=2, y_range_name="speed", color=Viridis4[0], legend_label = "True Airspeed [KAS]")
        pAltitude.extra_y_ranges.renderers = [kiasLine, tasLine]    #this does not quite work: https://stackoverflow.com/questions/48631530/bokeh-twin-axes-with-datarange1d-not-well-scaling
        pAltitude.y_range.renderers = [altitudeLine]

        # # Presented state
        # pState = figure(plot_width=1157, plot_height=300, x_range=pElev.x_range)
        # # Setting the second y axis range name and range
        # norm_state_extents = 10
        # pState.extra_y_ranges = {"normalized_data": Range1d(start=-norm_state_extents, end=norm_state_extents )}
        # # Adding the second axis to the plot.  
        # pState.add_layout(LinearAxis(y_range_name="normalized_data", axis_label="normalized data"), 'right')
        # state_lines = []
        # state_legend = []
        # normalized_state_lines = []
        # if self.presented_state:
        #     for idx, state_name in enumerate(self.presented_state):
        #         if(data_frame[state_name].max() <= norm_state_extents and data_frame[state_name].min() >= -norm_state_extents):
        #             normalized_state_lines.append(
        #                 pState.line(data_frame.index*self.step_time, data_frame[state_name], line_width=2, y_range_name="normalized_data", color=Inferno7[idx%7], visible=False)
        #             )
        #             state_legend.append( ("norm_"+state_name, [normalized_state_lines[-1]]) )                    
        #         else:     
        #             state_lines.append(
        #                 pState.line(data_frame.index*self.step_time, data_frame[state_name], line_width=2, color=Inferno7[idx%7], visible=False)
        #             )
        #             state_legend.append( (state_name, [state_lines[-1]]) )                    
        #     pState.y_range.renderers = state_lines
        #     pState.extra_y_ranges.renderers = normalized_state_lines    #this does not quite work: https://stackoverflow.com/questions/48631530/bokeh-twin-axes-with-datarange1d-not-well-scaling

        # lg_state = Legend(items = state_legend, location=(0, 0), glyph_width = 25, label_width = 290)
        # lg_state.click_policy="hide"
        # pState.add_layout(lg_state, 'right')

        # #Reward
        # pReward = figure(plot_width=1157, plot_height=300, x_range=pElev.x_range)
        # rwd_cmp_lines = []
        # reward_legend = []
        # rewardLine = pReward.line(data_frame.index*self.step_time, data_frame['reward'], line_width=2, color=Viridis4[3])
        # reward_legend.append( ("actual Reward", [rewardLine]) )
        # for idx, rwd_component in enumerate(self.reward_components_dict.keys()):
        #     rwd_cmp_lines.append (
        #         pReward.line(data_frame.index*self.step_time, data_frame[rwd_component], line_width=2, color=Viridis4[idx%4])
        #     )
        #     reward_legend.append( (rwd_component, [rwd_cmp_lines[-1]]) )
        # reward_lg = Legend(items = reward_legend, location=(48, 0), glyph_width = 25, label_width = 290)
        # reward_lg.click_policy="hide"
        # pReward.add_layout(reward_lg, 'right')


        tElev = Title()
        tElev.text = 'Flight Angle over Timesteps'
        pElev.title = tElev
        pElev.xaxis.axis_label = 'timestep [s]'
        pElev.yaxis[0].axis_label = 'Glide Path Angle [deg]'
        pElev.legend.click_policy="hide"

        tAil = Title()
        tAil.text = 'Roll Angle over Timesteps'
        pAileron.title = tAil
        pAileron.xaxis.axis_label = 'timestep [s]'
        pAileron.yaxis[0].axis_label = 'Roll Angle [deg]'
        pAileron.legend.click_policy="hide"

        tAlti = Title()
        tAlti.text = 'Altitude and Speed [IAS, TAS] over Timesteps'
        pAltitude.title = tAlti
        pAltitude.xaxis.axis_label = 'timestep [s]'
        pAltitude.yaxis[0].axis_label = 'Altitude [ftsl]'
        pAltitude.legend.location="center_right"
        pAltitude.legend.click_policy="hide"

        # tReward = Title()
        # tReward.text = 'actual Reward over Timesteps'
        # pReward.title = tReward
        # pReward.xaxis.axis_label = 'timestep [s]'
        # pReward.yaxis[0].axis_label = 'actual Reward [norm.]'


        # tState = Title()
        # tState.text = 'actual State presentation to Agent'
        # # pState.title = tReward
        # pState.xaxis.axis_label = 'timestep [s]'
        # pState.yaxis[0].axis_label = 'state data'


        #activate the zooming on all plots
        #this is not nice, but this not either: https://stackoverflow.com/questions/49282688/how-do-i-set-default-active-tools-for-a-bokeh-gridplot
        pElev.toolbar.active_scroll = pElev.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        pAileron.toolbar.active_scroll = pAileron.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        pAltitude.toolbar.active_scroll = pAltitude.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        # pReward.toolbar.active_scroll = pReward.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        # pState.toolbar.active_scroll = pState.toolbar.tools[1]    #this selects the WheelZoomTool instance 

        reset_output()

        # if self.env.meta_dict['model_type'] == 'trained':
        #     discriminator = self.env.meta_dict['model_base_name']+"_%+.2f" % (data_frame['reward'].sum())
        #     self.env.meta_dict['model_discriminator'] = discriminator
        # else: 
        #     discriminator = self.env.meta_dict['model_discriminator']

        grid = gridplot([[pElev, pAileron], [pAltitude, None]])
        #for string formatting look here: https://pyformat.info/

        titleString = "Run Date: {}; ".format(datetime.datetime.now().strftime("%c"))
        if 'episode_number' in self.env.meta_dict:
            titleString += "Episode: {}; ".format(self.env.meta_dict['episode_number'])
        # titleString += "Total Reward: {:.2f}; ".format(data_frame['reward'].sum())
        # titleString += "Model Discriminator: {};".format(self.env.meta_dict['model_discriminator'])
        webpage = column(
            Div(text="<h1>" + self.env.unwrapped.spec.id + 
            (" - " + self.env.meta_dict['env_info']) if 'env_info' in self.meta_dict else "" + 
            "</h1>"), 
            Div(text="<h2>"+titleString+"</h2>"), 
            grid)

        html_output_name = os.path.join(self.dirname, 'glideAngle_Elevator_latest.html')
        if self.showNextPlotFlag:
            output_file(html_output_name, mode='absolute') #use mode='absolute' to make it work offline with the js and css installed in the bokeh package locally
            if self.firstRun:
                show(webpage)  #opens up a new browser window
                self.firstRun = False
            else:
                save(webpage)  #just updates the HTML; Manual F5 in browser required :-(, (There must be a way to push...)

        
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

        
        

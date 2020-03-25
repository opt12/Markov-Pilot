import gym
import gym.spaces
import gym_jsbsim.properties as prp
import numpy as np
import pandas as pd
import datetime
import os
from bokeh.io import output_file, show, output_notebook, reset_output, save, export_png
import math

from timeit import timeit


class EpisodePlotterWrapper(gym.Wrapper):
    def __init__(self, env, presented_state = None):
        super(EpisodePlotterWrapper, self).__init__(env)

        self.step_frequency_hz = env.task.step_frequency_hz  #use this to scale the x-axis

        #create a pandas dataframe to hold all episode data
        # |state | reward | done | action| per each step with property names as data
        self.action_variables = env.task.action_variables
        self.state_variables = env.task.state_variables
        self.reward_variables = (prp.Property('reward', 'the reward dobtained in this step'),)
        self.done_variables = (prp.Property('done', 'indicates the end of an episode'),)
        self.state = np.empty(self.env.observation_space.shape)
        self.reward = 0.0
        self.done = False
        self.recorderData = self.state_variables + self.reward_variables + self.done_variables# + self.action_variables #actions are part of the state
        self.recorderCols = list(map(lambda el:el.get_legal_name(),self.recorderData))

        self.presented_state = presented_state

        self.showNextPlotFlag = False
        self.exportNextPlotFlag = False
        self.save_to_csv = False
        self.firstRun = True #to determine if we're supposed to open a new Browser window

        self.recorderDictList = []   #see https://stackoverflow.com/a/17496530/2682209

        self.plotCounter = 0    #this is just a counter that is incremented with every saved plot to chase the "too many open files" bug
        self.dirname = os.path.dirname(__file__) + '/../plots/{}'.format(datetime.datetime.now().strftime("%Y_%m_%d-%H:%M"))
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)



    def step(self, action):
        #let's move on to the next step
        self.newObs = self.env.step(action)
        self.state, self.reward, self.done, info = self.newObs
        self.reward_components_dict = info['reward_components']
        data = np.concatenate((self.state, [self.reward], [self.done], action)).tolist()
        dataDict = dict(zip(self.recorderCols, data))
        dataDict.update(self.reward_components_dict)
        self.recorderDictList.append(dataDict)
        if self.done:
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

        return (self.state, self.reward, self.done, info)

    def reset(self):
        self.recorderDictList = []   #see https://stackoverflow.com/a/17496530/2682209
        self.state = self.env.reset()
        #save the initial state
        data = np.concatenate((self.state, [0], [0], np.zeros(self.env.action_space.shape[0]))).tolist()
        dataDict = dict(zip(self.recorderCols, data))
        self.recorderDictList.append(dataDict)

        return self.state
    
    def showGraph(self,data_frame):
        from bokeh.plotting import figure
        from bokeh.layouts import row, column, gridplot
        from bokeh.io import output_file, show, output_notebook, reset_output, save, export_png
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

        elevatorLine = pElev.line(data_frame.index/self.step_frequency_hz, data_frame['fcs_elevator_cmd_norm'], line_width=1, y_range_name="elevator", color=Viridis4[1], legend_label = "Elevator Cmd.")
        gammaLine = pElev.line(data_frame.index/self.step_frequency_hz, data_frame['flight_path_gamma_deg'], line_width=2, color=Viridis4[0], legend_label="Path angle")
        targetGammaLine = pElev.line(data_frame.index/self.step_frequency_hz, data_frame['setpoint_glide_angle_deg'], line_width=2, color=Viridis4[3], legend_label="Target Path angle")
        aoaLine = pElev.line(data_frame.index/self.step_frequency_hz, data_frame['aero_alpha_deg'], line_width=1, color=Viridis4[2], legend_label="AoA")

        # RollAngle and Aileron
        pAileron = figure(plot_width=800, plot_height=400, x_range=pElev.x_range)
        # Setting the second y axis range name and range
        pAileron.extra_y_ranges = {"aileron": Range1d(start=-1, end=1)}
        # Adding the second axis to the plot.  
        pAileron.add_layout(LinearAxis(y_range_name="aileron", axis_label="Aileron Cmd [norm.]"), 'right')

        aileronLine = pAileron.line(data_frame.index/self.step_frequency_hz, data_frame['fcs_aileron_cmd_norm'], line_width=1, y_range_name="aileron", color=Viridis4[1], legend_label = "Aileron Cmd.")
        deltaAileronLine = pAileron.line(data_frame.index/self.step_frequency_hz, data_frame['info_delta_cmd_aileron'], line_width=1, y_range_name="aileron", color=Viridis4[2], legend_label = "Δ Ail. Cmd.")
        phiLine = pAileron.line(data_frame.index/self.step_frequency_hz, data_frame['attitude_phi_deg'], line_width=2, color=Viridis4[0], legend_label="Roll angle")
        targetPhiLine = pAileron.line(data_frame.index/self.step_frequency_hz, data_frame['setpoint_roll_angle_deg'], line_width=2, color=Viridis4[3], legend_label="Target Roll angle")
        

        #Altitude over ground
        pAltitude = figure(plot_width=800, plot_height=300, x_range=pElev.x_range)
        # Setting the second y axis range name and range
        pAltitude.extra_y_ranges = {"speed": Range1d(50, 120)}
        # Adding the second axis to the plot.  
        pAltitude.add_layout(LinearAxis(y_range_name="speed", axis_label="IAS, TAS [Knots]"), 'right')

        altitudeLine = pAltitude.line(data_frame.index/self.step_frequency_hz, data_frame['position_h_sl_ft'], line_width=2, color=Viridis4[2], legend_label = "Altitude [ftsl]")
        kiasLine = pAltitude.line(data_frame.index/self.step_frequency_hz, data_frame['velocities_vc_kts'], line_width=2, y_range_name="speed", color=Viridis4[1], legend_label = "Indicated Airspeed [KIAS]")
        tasLine = pAltitude.line(data_frame.index/self.step_frequency_hz, data_frame['velocities_vtrue_kts'], line_width=2, y_range_name="speed", color=Viridis4[0], legend_label = "True Airspeed [KAS]")
        pAltitude.extra_y_ranges.renderers = [kiasLine, tasLine]    #this does not quite work: https://stackoverflow.com/questions/48631530/bokeh-twin-axes-with-datarange1d-not-well-scaling
        pAltitude.y_range.renderers = [altitudeLine]

        # Presented state
        pState = figure(plot_width=1200, plot_height=300, x_range=pElev.x_range)
        # Setting the second y axis range name and range
        norm_state_extents = 10
        pState.extra_y_ranges = {"normalized_data": Range1d(start=-norm_state_extents, end=norm_state_extents )}
        # Adding the second axis to the plot.  
        pState.add_layout(LinearAxis(y_range_name="normalized_data", axis_label="normalized data"), 'right')
        state_lines = []
        state_legend = []
        normalized_state_lines = []
        if self.presented_state:
            for idx, state_name in enumerate(self.presented_state):
                if(data_frame[state_name].max() <= norm_state_extents and data_frame[state_name].min() >= -norm_state_extents):
                    normalized_state_lines.append(
                        pState.line(data_frame.index/self.step_frequency_hz, data_frame[state_name], line_width=2, y_range_name="normalized_data", color=Inferno7[idx%7])
                    )
                    state_legend.append( ("norm_"+state_name, [normalized_state_lines[-1]]) )                    
                else:     
                    state_lines.append(
                        pState.line(data_frame.index/self.step_frequency_hz, data_frame[state_name], line_width=2, color=Inferno7[idx%7])
                    )
                    state_legend.append( (state_name, [state_lines[-1]]) )                    
            pState.y_range.renderers = state_lines
            pState.extra_y_ranges.renderers = normalized_state_lines    #this does not quite work: https://stackoverflow.com/questions/48631530/bokeh-twin-axes-with-datarange1d-not-well-scaling

        lg_state = Legend(items = state_legend, location=(0, 0), glyph_width = 25, label_width = 333)
        lg_state.click_policy="hide"
        pState.add_layout(lg_state, 'right')

        #Reward
        pReward = figure(plot_width=1200, plot_height=300, x_range=pElev.x_range)
        rwd_cmp_lines = []
        reward_legend = []
        rewardLine = pReward.line(data_frame.index/self.step_frequency_hz, data_frame['reward'], line_width=2, color=Viridis4[3])
        reward_legend.append( ("actual Reward", [rewardLine]) )
        for idx, rwd_component in enumerate(self.reward_components_dict.keys()):
            rwd_cmp_lines.append (
                pReward.line(data_frame.index/self.step_frequency_hz, data_frame[rwd_component], line_width=2, color=Viridis4[idx%4])
            )
            reward_legend.append( (rwd_component, [rwd_cmp_lines[-1]]) )
        reward_lg = Legend(items = reward_legend, location=(48, 0), glyph_width = 25, label_width = 333)
        reward_lg.click_policy="hide"
        pReward.add_layout(reward_lg, 'right')


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

        tReward = Title()
        tReward.text = 'actual Reward over Timesteps'
        pReward.title = tReward
        pReward.xaxis.axis_label = 'timestep [s]'
        pReward.yaxis[0].axis_label = 'actual Reward [norm.]'


        tState = Title()
        tState.text = 'actual State presentation to Agent'
        # pState.title = tReward
        pState.xaxis.axis_label = 'timestep [s]'
        pState.yaxis[0].axis_label = 'state data'


        #activate the zooming on all plots
        #this is not nice, but this not either: https://stackoverflow.com/questions/49282688/how-do-i-set-default-active-tools-for-a-bokeh-gridplot
        pElev.toolbar.active_scroll = pElev.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        pAileron.toolbar.active_scroll = pAileron.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        pAltitude.toolbar.active_scroll = pAltitude.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        pReward.toolbar.active_scroll = pReward.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        pState.toolbar.active_scroll = pState.toolbar.tools[1]    #this selects the WheelZoomTool instance 

        reset_output()
        grid = gridplot([[pElev, pAileron], [pAltitude, pReward], [None, pState]])
        #for string formatting look here: https://pyformat.info/
        titleString = "Run Plot: {}; Total Reward: {:.2f}".format(datetime.datetime.now().strftime("%H:%M:%S"), data_frame['reward'].sum())
        webpage = column(Div(text="<h2>"+titleString+"</h2>"), grid)

        if self.showNextPlotFlag:
            self.plotCounter += 1   #increment the plot counter
            output_file(os.path.join(self.dirname, 'glideAngle_Elevator.html'), mode='inline') #use mode='inline' to make it work offline
            if self.firstRun:
                # placing this output_file here is a try to avoid the "too many open files exception"
                # output_file(os.path.join(self.dirname, '../plots/glideAngle_Elevator.html')) #use mode='inline' to make it work offline
                show(webpage)  #opens up a new browser window
                self.firstRun = False
            else:
                try:
                    save(webpage)  #just updates the HTML; Manual F5 in browser required :-(, (There must be a way to push...)
                except OSError as err:
                    #TODO: after a while it says too many open files exception. So I have to close that somehow
                    # This happens in bokeh.io.saving.py _save_helper() when opening the file with "with io.open(...) as fd"
                    print("OSError occurred after {} open files. Why this?".format(self.plotCounter))
                    print(err)


        
        if self.exportNextPlotFlag:
            # @timeit   TODO: die sourcen werden nicht gefunden
            def export(webpage):
                filename = os.path.join(self.dirname, 'glideAngle_Elevator_{}_Reward_{:.2f}.png'.format(datetime.datetime.now().strftime("%H:%M:%S"), data_frame['reward'].sum()))
                export_png(webpage, filename)
            export(webpage)

        self.showNextPlotFlag = False   #only show the plot once and then reset
        self.exportNextPlotFlag = False
        print("Output Plot generated: "+titleString)

    def showNextPlot(self, show = False, export = False, save_to_csv = False):
        self.showNextPlotFlag = show
        self.exportNextPlotFlag = export
        self.save_to_csv = save_to_csv
        
        

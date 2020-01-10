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
    def __init__(self, env):
        super(EpisodePlotterWrapper, self).__init__(env)

        #create a pandas dataframe to hold all episode data
        # |state | reward | done | action| per each step with property names as data
        self.action_variables = env.task.action_variables
        self.state_variables = env.task.state_variables
        self.reward_variables = (prp.Property('reward', 'the reward dobtained in this step'),)
        self.done_variables = (prp.Property('done', 'indicates the end of an episode'),)
        self.state = np.empty(self.env.observation_space.shape)
        self.reward = 0.0
        self.done = False
        self.recorderData = self.state_variables + self.reward_variables + self.done_variables + self.action_variables
        self.recorderCols = list(map(lambda el:el.get_legal_name(),self.recorderData))

        self.showNextPlotFlag = False
        self.exportNextPlotFlag = False
        self.firstRun = True #to determine if we're supposed to open a new Browser window

        self.recorderDictList = []   #see https://stackoverflow.com/a/17496530/2682209

        self.plotCounter = 0    #this is just a counter that is incremented with every saved plot to chase the "too many open files" bug

    def step(self, action):
        #save the action and the sate and reward before
        data = np.concatenate((self.state, [self.reward], [self.done], action)).tolist()
        dataDict = dict(zip(self.recorderCols, data))
        self.recorderDictList.append(dataDict)
        #now let's move on to the next step
        self.newObs = self.env.step(action)
        self.state, self.reward, self.done, info = self.newObs
        if self.done:
            contains_nan = any(math.isnan(el) for el in self.state)
            if (contains_nan or 
                    self.showNextPlotFlag or self.exportNextPlotFlag):
                data = np.concatenate((self.state, [self.reward], [self.done], np.full(self.env.action_space.shape, np.nan))).tolist()
                dataDict = dict(zip(self.recorderCols, data))
                self.recorderDictList.append(dataDict)
                dataRecorder = pd.DataFrame(self.recorderDictList)    
                if (contains_nan):
                    #save the entire pandas frame to CSV file
                    dirname = os.path.dirname(__file__)
                    filename = os.path.join(dirname, '../nan/state_with_NaN_{}.csv'.format(datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")))
                    dataRecorder.to_csv(filename)
                # print(dataRecorder.keys())   #this is handy if you want to change the plot to get the available data headings
                self.showGraph(dataRecorder)

        return (self.state, self.reward, self.done, info)

    def reset(self):
        self.recorderDictList = []   #see https://stackoverflow.com/a/17496530/2682209
        self.state = self.env.reset()
        return self.state
    
    def showGraph(self,df):
        from bokeh.plotting import figure
        from bokeh.layouts import row, column, gridplot
        from bokeh.io import output_file, show, output_notebook, reset_output, save, export_png
        from bokeh.models.annotations import Title, Legend
        from bokeh.models.widgets.markups import Div
        from bokeh.models import LinearAxis, Range1d
        from bokeh.palettes import Viridis4

        # GlideAngle and Elevator
        pElev = figure(plot_width=800, plot_height=500)
        # Setting the second y axis range name and range
        pElev.extra_y_ranges = {"elevator": Range1d(start=-1, end=1)}
        # Adding the second axis to the plot.  
        pElev.add_layout(LinearAxis(y_range_name="elevator", axis_label="Elevator Cmd [norm.]"), 'right')

        elevatorLine = pElev.line(df.index, df['fcs_elevator_cmd_norm'], line_width=1, y_range_name="elevator", color=Viridis4[1], legend_label = "Elevator Cmd.")
        gammaLine = pElev.line(df.index, df['flight_path_gamma_deg'], line_width=2, color=Viridis4[0], legend_label="Path angle")

        # RollAngle and Aileron
        pAileron = figure(plot_width=800, plot_height=500, x_range=pElev.x_range)
        # Setting the second y axis range name and range
        pAileron.extra_y_ranges = {"aileron": Range1d(start=-1, end=1)}
        # Adding the second axis to the plot.  
        pAileron.add_layout(LinearAxis(y_range_name="aileron", axis_label="Aileron Cmd [norm.]"), 'right')

        aileronLine = pAileron.line(df.index, df['fcs_aileron_cmd_norm'], line_width=1, y_range_name="aileron", color=Viridis4[1], legend_label = "Aileron Cmd.")
        phiLine = pAileron.line(df.index, df['attitude_phi_deg'], line_width=2, color=Viridis4[0], legend_label="Roll angle")

        #Altitude over ground
        pAltitude = figure(plot_width=800, plot_height=300, x_range=pElev.x_range)
        # Setting the second y axis range name and range
        pAltitude.extra_y_ranges = {"speed": Range1d(50, 120)}
        # Adding the second axis to the plot.  
        pAltitude.add_layout(LinearAxis(y_range_name="speed", axis_label="IAS, TAS [Knots]"), 'right')

        altitudeLine = pAltitude.line(df.index, df['position_h_sl_ft'], line_width=2, color=Viridis4[2], legend_label = "Altitude [ftsl]")
        kiasLine = pAltitude.line(df.index, df['velocities_vc_kts'], line_width=2, y_range_name="speed", color=Viridis4[1], legend_label = "Indicated Airspeed [KIAS]")
        tasLine = pAltitude.line(df.index, df['velocities_vtrue_kts'], line_width=2, y_range_name="speed", color=Viridis4[0], legend_label = "True Airspeed [KAS]")
        pAltitude.extra_y_ranges.renderers = [kiasLine, tasLine]    #this does not wuite work: https://stackoverflow.com/questions/48631530/bokeh-twin-axes-with-datarange1d-not-well-scaling
        pAltitude.y_range.renderers = [altitudeLine]

        #Reward
        pReward = figure(plot_width=800, plot_height=300, x_range=pElev.x_range)
        rewardLine = pReward.line(df.index, df['reward'], line_width=2, color=Viridis4[3], legend_label = "actual Reward")

        tElev = Title()
        tElev.text = 'Flight Angle over Timesteps'
        pElev.title = tElev
        pElev.xaxis.axis_label = 'timestep [0.2s]'
        pElev.yaxis[0].axis_label = 'Glide Path Angle [deg]'

        tAil = Title()
        tAil.text = 'Roll Angle over Timesteps'
        pAileron.title = tAil
        pAileron.xaxis.axis_label = 'timestep [0.2s]'
        pAileron.yaxis[0].axis_label = 'Roll Angle [deg]'

        tAlti = Title()
        tAlti.text = 'Altitude and Speed [IAS, TAS] over Timesteps'
        pAltitude.title = tAlti
        pAltitude.xaxis.axis_label = 'timestep [0.2s]'
        pAltitude.yaxis[0].axis_label = 'Altitude [ftsl]'

        tReward = Title()
        tReward.text = 'actual Reward over Timesteps'
        pReward.title = tReward
        pReward.xaxis.axis_label = 'timestep [0.2s]'
        pReward.yaxis[0].axis_label = 'actual Reward [norm.]'

        #activate the zooming on all plots
        #this is not nice, but this not either: https://stackoverflow.com/questions/49282688/how-do-i-set-default-active-tools-for-a-bokeh-gridplot
        pElev.toolbar.active_scroll = pElev.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        pAileron.toolbar.active_scroll = pAileron.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        pAltitude.toolbar.active_scroll = pAltitude.toolbar.tools[1]    #this selects the WheelZoomTool instance 
        pReward.toolbar.active_scroll = pReward.toolbar.tools[1]    #this selects the WheelZoomTool instance 

        reset_output()
        grid = gridplot([[pElev, pAileron], [pAltitude, pReward]])
        #for string formatting look here: https://pyformat.info/
        titleString = "Run Plot: {}; Total Reward: {:.2f}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), df['reward'].sum())
        webpage = column(Div(text="<h2>"+titleString+"</h2>"), grid)

        dirname = os.path.dirname(__file__)

        if self.showNextPlotFlag:
            self.plotCounter += 1   #increment the plot counter
            output_file(os.path.join(dirname, '../plots/glideAngle_Elevator.html'), mode='inline') #use mode='inline' to make it work offline
            if self.firstRun:
                # placing this output_file here is a try to avoid the "too many open files exception"
                # output_file(os.path.join(dirname, '../plots/glideAngle_Elevator.html')) #use mode='inline' to make it work offline
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
                filename = os.path.join(dirname, '../plots/glideAngle_Elevator_{}_Reward_{:.2f}.png'.format(datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S"), df['reward'].sum()))
                export_png(webpage, filename)
            export(webpage)

        self.showNextPlotFlag = False   #only show the plot once and then reset
        self.exportNextPlotFlag = False
        print("Output Plot generated: "+titleString)

    def showNextPlot(self, show = False, export = False):
        self.showNextPlotFlag = show
        self.exportNextPlotFlag = export
        
        

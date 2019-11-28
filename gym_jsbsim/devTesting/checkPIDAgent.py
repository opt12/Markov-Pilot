import sys            
sys.path.append(r'/home/felix/git/gym-jsbsim/')

import gym
from gym.wrappers import Monitor
import gym_jsbsim
from gym_jsbsim.agents import PIDAgent
import numpy as np
import gym_jsbsim.properties as prp

import time
import threading
import rpyc
import pandas as pd
# import altair as alt


class RPYCClient(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        conn = rpyc.connect("localhost", 12345)
        self.bgsrv = rpyc.BgServingThread(conn, self.backgroundStoppedCb)
        self.bgsrv.SLEEP_INTERVAL = 0.025   #to make the GUI more reactive
        self.bgsrvRunning = True
        self.c = conn.root
        # callback("Hallo aus der __init__")
        # self.registerCallbackBtn("from RPYCCLient", callback)

    def backgroundStoppedCb(self):
        self.bgsrvRunning = False

    def addQuadSlider(self, name, setMin = 15, setMax = -15, setRes = 0.2, cb = None, **options):
        self.c.addQuadSlider(name, setMin, setMax, setRes, cb, **options)
    
    def printMessage(self, msg):
        self.c.printMessage(msg)
    
    def setValue(self, QuadSliderKey, SliderKey, value):
        return self.c.setValue(QuadSliderKey, SliderKey, value)

    def run(self):
        while self.bgsrvRunning:
            time.sleep(1)

def getCallback(env, agent, name, propName, inverted=False, **kwargs):
    def tuningCallback(**kwargs):
        print("tuning: {}:".format(name))
        for key, value in kwargs.items(): 
            print ("%s == %s" %(key, value)) 
        #change PID params
        if inverted:
            agent.setControllerParams(name, -kwargs['valueP'], -kwargs['valuePI'], -kwargs['valuePD'], kwargs['valueEnabled'])
        else:
            agent.setControllerParams(name, kwargs['valueP'], kwargs['valuePI'], kwargs['valuePD'], kwargs['valueEnabled'])
        #change setpoint
        env.sim.jsbsim[propName] = kwargs['valueSetPoint']

    return tuningCallback

def showGraph(df):
    from bokeh.plotting import figure
    from bokeh.layouts import row, column, gridplot
    from bokeh.io import output_file, show, output_notebook
    from bokeh.models.annotations import Title, Legend
    from bokeh.models import LinearAxis, Range1d
    from bokeh.palettes import Viridis4

    # GlideAngle and Elevator
    pElev = figure(plot_width=800, plot_height=600)
    # Setting the second y axis range name and range
    pElev.extra_y_ranges = {"elevator": Range1d(start=-1, end=1)}
    # Adding the second axis to the plot.  
    pElev.add_layout(LinearAxis(y_range_name="elevator", axis_label="Elevator Cmd [norm.]"), 'right')

    gammaLine = pElev.line(df.index, df['flight_path_gamma_deg'], line_width=2, color=Viridis4[0], legend_label="Path angle")
    elevatorLine = pElev.line(df.index, df['fcs_elevator_cmd_norm'], line_width=2, y_range_name="elevator", color=Viridis4[1], legend_label = "Elevator Cmd.")

    # RollAngle and Aileron
    pAileron = figure(plot_width=800, plot_height=600, x_range=pElev.x_range)
    # Setting the second y axis range name and range
    pAileron.extra_y_ranges = {"aileron": Range1d(start=-1, end=1)}
    # Adding the second axis to the plot.  
    pAileron.add_layout(LinearAxis(y_range_name="aileron", axis_label="Aileron Cmd [norm.]"), 'right')

    phiLine = pAileron.line(df.index, df['attitude_phi_deg'], line_width=2, color=Viridis4[0], legend_label="Roll angle")
    aileronLine = pAileron.line(df.index, df['fcs_aileron_cmd_norm'], line_width=2, y_range_name="aileron", color=Viridis4[1], legend_label = "Aileron Cmd.")

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


    output_file('glideAngle_Elevator.html')
    grid = gridplot([[pElev, pAileron], [pAltitude, pReward]])
    show(grid)


if __name__ == "__main__":
    env = gym.make("JSBSim-SteadyGlideTask-Cessna172P-Shaping.STANDARD-FG-v0")
    state = env.reset()
    total_reward = 0.0
    done = None
    reward = None
    sim = env.sim
    agent_interaction_freq = env.JSBSIM_DT_HZ / env.sim_steps_per_agent_step
    agent = PIDAgent(env.action_space, agent_interaction_freq)
    # agent = ConstantAgent(env.action_space)
    action = np.empty(env.action_space.shape)
    #create a pandas dataframe to hold all episode data
    # |state | reward | done | action| per each step with property names as data
    action_variables = env.task.action_variables
    state_variables = env.task.state_variables
    reward_variables = (prp.Property('reward', 'the rewar dobtained in this step'),)
    done_variables = (prp.Property('done', 'indicates the end of an episode'),)
    recorderData = state_variables + reward_variables + done_variables + action_variables
    recorderCols = list(map(lambda el:el.get_legal_name(),recorderData))

    recorderDictList = []   #see https://stackoverflow.com/a/17496530/2682209

    try:
        client = RPYCClient()
        client.start()
        print("started client")
        try:
            client.addQuadSlider(name="pitchControl", 
                    cb = getCallback(env, agent, 'pitchControl', 'target/glideAngle-deg', inverted = True))
            client.setValue('pitchControl', 'sliderP', 5e-2)   #TODO: it's only positive so far
            client.setValue('pitchControl', 'sliderPI', 6.5e-2)   #TODO: it's only positive so far
            client.setValue('pitchControl', 'sliderPD', 1e-3)   #TODO: it's only positive so far
            client.setValue('pitchControl', 'sliderSetpoint', -1)   #TODO: it's only positive so far
            client.addQuadSlider(name="rollControl", setMin = 35, setMax = -35, setRes = 1, 
                    cb = getCallback(env, agent, 'rollControl', 'target/roll-deg'))
            client.setValue('rollControl', 'sliderP', 2e-2)   #TODO: it's only positive so far
            client.setValue('rollControl', 'sliderPI', 2e-2)   #TODO: it's only positive so far
            client.setValue('rollControl', 'sliderPD', 0)   #TODO: it's only positive so far
            client.setValue('rollControl', 'sliderSetpoint', 15)
        except:
            print("could not register controller widgets")
    except:
        print("GUI-Client did not start successfully. Sorry")
    print("prepared!")

    frame_idx = 0
    ts_frame = 0
    ts = time.time()

    while True:
        frame_idx += 1
#        action = env.action_space.sample()
        action = agent.act(state)
        #save the action and the sate and reward before
        data = np.concatenate((state, [reward], [done], action)).tolist()
        dataDict = dict(zip(recorderCols, data))
        recorderDictList.append(dataDict)
        state, reward, done, _ = env.step(action)
        # env.render('human')
        # env.render('timeline')
        # env.render('flightgear')
        # print("Action: {}; Observation: {}".format(action, obs))
        total_reward += reward
        # if frame_idx%10 == 0:
        #     print(frame_idx)
        if done:
            data = np.concatenate((state, [reward], [done], np.empty(env.action_space.shape))).tolist()
            dataDict = dict(zip(recorderCols, data))
            recorderDictList.append(dataDict)
            break

    runtime = (time.time() - ts)
    speed = (frame_idx - ts_frame) / runtime
    print("%d frames in %f seconds: %f fps"% (frame_idx, runtime, speed))
    dataRecorder = pd.DataFrame(recorderDictList)    
    print(dataRecorder.keys())   

    showGraph(dataRecorder)
    


    # chartFlightPath = alt.Chart(dataRecorder.reset_index()).mark_line().encode(
    #     x='index',
    #     y='flight_path_gamma_deg'
    # )
    # chartElevator = alt.Chart(dataRecorder.reset_index()).mark_line(color='green').encode(
    #     x='index',
    #     y='fcs_elevator_cmd_norm'
    # )
    # chart = chartFlightPath + chartElevator
    # chart = chart.resolve_scale(y='independent')
    # chart.serve()

    # chartRoll = alt.Chart(dataRecorder.reset_index()).mark_line().encode(
    #     x='index',
    #     y='attitude_phi_deg'
    # )
    # chartAileron = alt.Chart(dataRecorder.reset_index()).mark_line(color='green').encode(
    #     x='index',
    #     y='fcs_aileron_cmd_norm'
    # )
    # chartRollAngle = chartRoll + chartAileron
    # chartRollAngle = chartRollAngle.resolve_scale(y='independent')
    # chartRollAngle.serve()
    print("Reward got: %.2f" % total_reward)

import time
import threading
import rpyc
import tkinter as tk
from  tkinter import *

number = 1


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

def callback(**kwargs):  
    for key, value in kwargs.items(): 
        print ("%s == %s" %(key, value)) 


if __name__ == '__main__':

    try:
        client = RPYCClient()
        client.start()
    except:
        print("Client did not start successfully. Sorry")
    
    print("started client")
    client.printMessage("Hallo")
    time.sleep(1)
    try:
        client.addQuadSlider(name="Pitch-Control", cb = callback)
    except:
        print("could not register Pitch-control")

    client.setValue('Pitch-Control', 'sliderSetpoint', -6)

    print("All done!")

    client.join()
    print("RPyC thread closed")

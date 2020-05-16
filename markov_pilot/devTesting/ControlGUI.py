import tkinter as tk
import tkinter.ttk as ttk
import math as m

# main definitions
import time
# rpyc servic definition
import rpyc

class MyService(rpyc.Service):
    def __init__(self, tkRoot):
        rpyc.Service.__init__(self)
        self.root = tkRoot
        print('Instantiated MyService with root: {}'.format(self.root))
    
    def exposed_addQuadSlider(self, name, setMin = 15, setMax = -15, setRes = 0.2, cb = None, **options):
        return self.root.addQuadSlider(name=name, setMin= setMin, setMax=setMax, setRes = setRes, cb = cb, **options)
    def exposed_printMessage(self, msg):
        return self.root.printMsg(msg)
    def exposed_setValue(self,QuadSliderKey, SliderKey, value):
        return self.root.setValue(QuadSliderKey, SliderKey, value)
    
class SliderWithExponent(ttk.Frame):
    def __init__(self, master, name, initialVal=50, **options):    
        self.minSlider = 0
        self.maxSlider = 10
        self.minExp = -5
        self.maxExp = 0
        ttk.Frame.__init__(self, master, **options,)
        self.sliderVar = tk.DoubleVar()
        self.slider = tk.Scale(self, name="scale", orient=tk.HORIZONTAL, length = 250, 
                from_=0, to=10, variable = self.sliderVar, command= self.sliderCb, takefocus=True)
        self.slider.config(resolution = 0.05)
        # with Windows OS
        self.slider.bind("<MouseWheel>", self.mouse_wheel)
        # with Linux OS
        self.slider.bind("<Button-4>", self.mouse_wheel)
        self.slider.bind("<Button-5>", self.mouse_wheel)

        self.exponent = tk.Spinbox(self, name="spinbox", 
                from_=-5, to=0, increment = 1, command= self.sliderCb)
        # with Windows OS
        self.exponent.bind("<MouseWheel>", self.mouse_wheel)
        # with Linux OS
        self.exponent.bind("<Button-4>", self.mouse_wheel)
        self.exponent.bind("<Button-5>", self.mouse_wheel)

        self.set(initialVal)

        self.label = ttk.Label(self, name="label",)
        self.value = tk.DoubleVar()
        separator = ttk.Frame(self)
        ttk.Label(separator, text = name, width = 10).grid(row=0, column=0)
        ttk.Separator(separator, orient=tk.HORIZONTAL, ).grid(row=0, column=1, sticky=tk.E+tk.W)
        separator.pack()
        self.slider.pack(fill='x', side='left')
        self.exponent.pack(fill='x', side='left')
        self.label.pack(fill='x', side='left')
        self.sliderCb()
    
    def sliderCb(self, event=None):
        sliderVal = self.children["scale"].get()
        try:
            expVal = int(self.children["spinbox"].get())
            value = sliderVal * 10**expVal
            self.value.set(value)
            self.label.config(text="{:.2e}".format(value))
        except:
            return #there is a race condition: self.children["spinbox"].get() can yield '' in case it is changed from somewhere else
    
    def mouse_wheel(self, event):
        source = event.widget
        if isinstance(source, tk.Spinbox):
            # respond to Linux or Windows wheel event
            if event.num == 5 or event.delta == -120:
                source.invoke('buttondown')
            if event.num == 4 or event.delta == 120:
                source.invoke('buttonup')
            return
        elif isinstance(source, tk.Scale):
            val = float(source.get())
            # respond to Linux or Windows wheel event
            if event.num == 5 or event.delta == -120:
                val -= source.cget('resolution')
            if event.num == 4 or event.delta == 120:
                val += source.cget('resolution')
            source.set(val)
            return        

    def set(self, newVal):
        # print("set slider to {}; min: {}, max{}".format(newVal, self.minSlider*10**self.minExp, self.maxSlider*10**self.maxExp))
        if newVal <= self.minSlider*10**self.minExp:
            self.sliderVar.set(self.minSlider)
            self.exponent.delete(0,"end")
            self.exponent.insert(0,self.minExp)
            self.sliderCb()
            return
        if newVal >= self.maxSlider*10**self.maxExp:
            self.sliderVar.set(self.maxSlider)
            self.exponent.delete(0,"end")
            self.exponent.insert(0,self.maxExp)
            self.sliderCb()
            return
        exp = m.floor(m.log10(newVal))
        valNorm = newVal / 10**exp
        self.sliderVar.set(valNorm)
        self.exponent.delete(0,"end")
        self.exponent.insert(0,exp)
        self.sliderCb()
        return



class QuadrupleSliderWithEnable(ttk.Frame):
    def __init__(self, master, name, setMin = 15, setMax = -15, setRes = 0.2, cb = None, **options):
        self.cb = cb
        tk.Frame.__init__(self, master, **options)
        sliderFrame = tk.Frame(self)
        ttk.Label(sliderFrame, text=name).pack()
        self.sliders = {
            "sliderP" : SliderWithExponent(sliderFrame, "SliderP"),
            "sliderPI" : SliderWithExponent(sliderFrame, "SliderPI"),
            "sliderPD" : SliderWithExponent(sliderFrame, "SliderPD"),
        }
        self.valueP = self.sliders["sliderP"].value
        self.valuePI = self.sliders["sliderPI"].value
        self.valuePD = self.sliders["sliderPD"].value
        self.valueP.trace('w', self.varChange)
        self.valuePI.trace('w', self.varChange)
        self.valuePD.trace('w', self.varChange)
        self.sliders["sliderP"].pack()
        self.sliders["sliderPI"].pack()
        self.sliders["sliderPD"].pack()

        SetpointEnableFrame = tk.Frame(self)
        quitBtn = tk.Button(self, text='Remove Quad', command=self.destroy) #TODO: it's not deleted from the widget list :-(
        self.setpointVar = tk.DoubleVar(self)
        self.sliders["sliderSetpoint"] = tk.Scale(SetpointEnableFrame, name="scale", orient=tk.VERTICAL,
                from_=setMin, to=setMax, resolution = setRes, takefocus=True, 
                command= self.sliderCb)
        # with Windows OS
        self.sliders["sliderSetpoint"].bind("<MouseWheel>", self.mouse_wheel)
        # with Linux OS
        self.sliders["sliderSetpoint"].bind("<Button-4>", self.mouse_wheel)
        self.sliders["sliderSetpoint"].bind("<Button-5>", self.mouse_wheel)
        self.enabledVar = tk.BooleanVar(self)
        enableBtn = ttk.Checkbutton(SetpointEnableFrame, text = "enabled", variable = self.enabledVar, command = None)
        self.enabledVar.set(True)

        self.setpointVar.trace('w', self.varChange)
        self.enabledVar.trace('w', self.varChange)

        self.sliders["sliderSetpoint"].pack()
        enableBtn.pack()
        quitBtn.pack()

        sliderFrame.pack(fill='x', side='left')
        SetpointEnableFrame.pack(fill='x', side='left')
    
    def mouse_wheel(self, event):
        source = event.widget
        val = float(source.get())
        # respond to Linux or Windows wheel event
        if event.num == 5 or event.delta == -120:
            val -= source.cget('resolution')
        if event.num == 4 or event.delta == 120:
            val += source.cget('resolution')
        source.set(val)
        return      

    def sliderCb(self, event=None):
        sliderVal = self.sliders["sliderSetpoint"].get()
        self.setpointVar.set(sliderVal)


    def set(self, element, value):
        self.sliders[element].set(value)
        self.varChange()

    def varChange(self, *argv):
        dict = {
            'valueP'  : self.valueP.get(), 
            'valuePI' : self.valuePI.get(), 
            'valuePD' : self.valuePD.get(), 
            'valueSetPoint' : self.setpointVar.get(),
            'valueEnabled' : self.enabledVar.get()
        }
        if self.cb:
            self.cb(**dict)
    



class GUI():
    def __init__(self):
        self.widgetList = {}
        self.root = tk.Tk()

        #now show the container
        self.root.title('This is my frame for all and everything')
        self.root.geometry("500x500")
        tk.Button(self.root, text="Finish", 
            command=lambda : self.root.event_generate("<Destroy>", when="now")).pack(anchor=tk.CENTER)
        # self.root.after(2000, self.scheduledAdd)

    def showGui(self):
        self.root.mainloop()

    def addQuadSlider(self, name, setMin = 15, setMax = -15, setRes = 0.2, cb = None, **options):
        newFrame = QuadrupleSliderWithEnable(None, name=name, setMin= setMin, setMax=setMax, setRes = setRes, cb = cb, **options)
        self.widgetList[name] = newFrame
        newFrame.pack()
    
    def setValue(self, QuadSliderKey, SliderKey, value):
        self.widgetList[QuadSliderKey].set(SliderKey, value)

    def printMsg(self, msg):
        print(msg)
        return 42
    

def callback(**kwargs):  
    for key, value in kwargs.items(): 
        print ("%s == %s" %(key, value)) 

# the main logic
if __name__ == '__main__':
    print("start")
    guiRoot = GUI()

    # guiRoot.addQuadSlider("Pitch-Control", cb = callback)
    # guiRoot.addQuadSlider("Roll-Control", cb = callback, setMin=30, setMax=-30, setRes=1)

    # guiRoot.widgetList["Pitch-Control"].set('sliderSetpoint', 20)
    # guiRoot.widgetList["Pitch-Control"].set('sliderPI', 5)
    # guiRoot.widgetList["Pitch-Control"].set('sliderPD', 0.00345)
    # guiRoot.widgetList["Pitch-Control"].set('sliderP', 0.0125)

    # start the rpyc server
    from rpyc.utils.server import ThreadedServer
    from threading import Thread
    myServiceInstance = MyService(guiRoot)
    server = ThreadedServer(myServiceInstance, port = 12345)
    t = Thread(target = server.start)
    t.daemon = True
    t.start()

    guiRoot.showGui()

    print("Server shutdown")

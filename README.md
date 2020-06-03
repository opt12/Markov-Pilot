# Markov-Pilot

This is the software developed in the course of a Master's thesis at Fernuniversität Hagen in 2020 with the title:

**Learning to Fly -- Building an Autopilot System based on Neural Networks and Reinforcement Learning**

The software contains an implementation of a Multi-Agent Reinforcement learning environment together with appropriate DDPG and MADDPG agents.

It is based on the [groundwork of Gor-Ren](https://github.com/Gor-Ren/gym-jsbsim), but has evolved a lot since forking his repo.

The software and its usage is documented in chapter 5 of the thesis which can be found in the [thesis](https://github.com/opt12/Markov-Pilot/tree/master/thesis) directory of this repo.



This work contributes to the final goal of building an autopilot system based on artificial neural networks.
Firstly, an overview is given on the state of the art of reinforcement learning in continuous spaces and the deep deterministic policy gradient (DDPG) algorithm utilized in this work. This is followed by reasoning about the application of reinforcement learning techniques on aircraft control and the formulation of continuous control tasks as Markov decision processes respectively Markov games. Based on this theory, a flexible software framework for experimentation is implemented that supports the definition of multiple tasks in a simulated aircraft environment with multiple reinforcement learning agents. Eventually, experiments were conducted using this software to determine a suitable reward structure for the flight control task definition. Several 3-axes flight controllers were trained using different algorithmic settings. The results are compared with conventional PID control, which was outperformed by one of the trained controllers. A results summary and an outlook on future research desiderata conclude this work.



## Dependencies

* [JSBSim](https://github.com/JSBSim-Team/jsbsim) flight dynamics model, including the C++ and Python libraries; It's crucial to have the patch from this issue available in your version of JSBSim: https://github.com/JSBSim-Team/jsbsim/issues/201
* FlightGear simulator (optional for visualisation)
* numpy, gym, matplotlib, pandas, bokeh, pytorch, ...

(There is supposedly something missing in the dependencies... I'll fix that, when I fix the setup.py)

## Installation
Firstly, follow the instructions on the [JSBSim](https://github.com/JSBSim-Team/jsbsim) repository to install JSBSim and its libraries.

Confirm that JSBSim is installed from the terminal:

```
$ JSBSim --version
JSBSim Version: 1.0.0 Jul 16 2018 09:14:35
```

and confirm that its Python library is correctly installed from a Python interpreter or IDE:

```
import jsbsim
```

The installation of the *Markov-Pilot* itself is still under construction. Sorry!


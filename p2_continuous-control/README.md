# Project 2: Continous control

![reacher_env](https://github.com/danny-bit/udacity_rl/assets/59084863/cf66f57e-818a-4d8a-95d1-ef9a6acf08dd)

### Introduction

The environment is a Unity environment called Reacher-v2.
The goal is to train an agent that can move a double-jointed robot arm to target locations.

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The task is episodic, and in order to solve the environment, the the average score of all agent must get an average score of +30 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Instructions

Follow `Continuous_Control.ipynb` to see how the agent was trained.
The learned model weights can be found in 
- checkpoint_actor.pth, checkpoint_critic.pth (latest solved environment weights)
- checkpoint_actor_best.pth, checkpoint_critic_best.pth (weights of the model with the highest average score over last 100 episodes)

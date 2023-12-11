# Project 3: Collaboration and Competition

![tennis_env](https://github.com/danny-bit/udacity_rl/assets/59084863/57210724-e4d1-42b9-b4a8-af4709d15d54)

### Introduction

The environment is a Unity environment called Tennis.
Is is an multi-agent environment, where two agents are trained to compete in Tennis.

An agent receives a reward of $0.1$ for a hit and a reward of $-0.01$ for a miss.
Thus each agent wants to keep the ball at play.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

3. Place the file in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Instructions

Have a look at the python script `main.py` to see how the agent was trained.
The result of the training can be plotted using `plotTraining.py`.
The learned model weights can be found in 
- checkpoint_actor_i.pth, checkpoint_critic_i.pth (latest solved environment weights)
- checkpoint_actor_besti.pth, checkpoint_critic_besti.pth (weights of the model with the highest average score over last 100 episodes)

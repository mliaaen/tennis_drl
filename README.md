[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the source folder, and unzip (or decompress) the file. Update main.py and Tennis.ipynb  if necssary with a different
path. For instance, if you are using a Mac, then you  downloaded Tennis.app. If this file is in the same folder as the notebook, then the line below should appear as follows:

    env = UnityEnvironment(file_name="Tennis.app")

### Installation Instructions

1. Create and activate a virtual python3 environment using virtualenv (or conda) 

 ``bash
pip3 install virtualenv (if not already done)
virtualenv -p python3 drlnd
source drlnd/bin/activate
```
    
2. Install the requirments


    
```bash
git clone https://github.com/mliaaen/tennis_drl.git
cd tennis_drl/python
 
pip install -r requiremnts.txt
python setup.py (install the unity agent)
cd ..

```
3. Follow the instructions in `Tennis.ipynb` to see how this environment behaves.
     
```bash
jupyter notebook Tennis.ipynd 

```   

The main.py contains the trainig code for this agent. Here you can select the parameters to use and start trining and plotting.
See the report.md for the report.

#### Training - generate configurations and train

Example: python gen_params.py --scenrios template.json --output myjobs.json 

This generates a json file with the configurations that shall be run.

Example: python train.py --config myjobs.json --output myresults

Will run all the congigurations and store the results in myresults folder. A summary
is found in scan_results.csv. Here you can find all the results and run times.

A run example is found here Tennis.ipynb


#### Running - using the trained agent

Check out the Tennis_Play.ipynb

The  last successful trained parameter set is store in models/checkpoint_actor_local_0/1.pth and
models/checkpoint_critic_local_0/1.pth




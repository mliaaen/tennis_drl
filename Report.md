
# The model

I ended up to use MADDPG for this problem since we have a continous action space.
It is also a multiagent problem since the agents have to be aware of the others actions
and observations during training. From 
a single agent the environment view this is non-stationary since it do not know what the other
agents does or will do. Using a simpler DDPG (as in preivous project) does not work (I tried) if 
you use only the local observations as input during training (and running). I have seen solutions to 
this problem using DDPG which uses all input and actions into training one (big) agent that controls 
both rackets, but this is not a multiagent solution.

The simplest approach was to build on the the DDPG model and
extend it to the MADDPG case. There are two agents (players) that needs to be trained and the during
training the agents will use the full oberservation space from both agents and the actions of the other
agents (here just one, but could be more) to train the critic network. The actor network will only 
use it own observation space as input. 

DDPG is an off-policy method that utilizes a replay buffer  for the training. This way it is possible 
to get stable training. When running the agent in real play only the actor is used to select the 
actions (two continious values) controlling the racket.

Both actor and critic networks uses both local and target networks that trains slowly using soft updates 
from the local to the target network in order to stabilize training.


# Trainig Results

In order to figure out how the MADDPG behaved during training, I created a little framwork for hyper-parameter searching.
It turned out to be quite useful and gave me insight into the behaviour of the model. I had 
had the trainig running for several days and keeping the results I could see the effects as the different paramteres changed.
Only one parameter change per round was done.

## Learning speed

The model was set up to learn every 5th round. Each time it does learning it updates 3 times. I tried with 10 times, but that 
just made the network stop learning. 

## Learning rates

The results showed that using a lower learning rate on the critic than the actor lead to faster
training. Why this is the case is a bit unclear. In all the good test runs the learning reate 
for the critic was lower or equal to the actor learning rate.

## Hidden layer size

I used two layer equal hidden layers in the nueral networks. I tried with 64, 96, 128 and 256 nodes. Using a
the bigger network only leady slower learning times and now performnace gains.


## Batch size

I tried batch sizes og 64, 128, 256 and 512, but it didnt make much difference in the training.
It seems to be an interesting correlation between Hidden layer size and batch size. The smallest batch size that worked was 128 
with a layer size of 64. 

## Seeding

I wanted to test out how different random seeds affected the training and it turned out that changing the 
random seed had big effect on the training time, but I did not see any runs where it did not train within my 2500 episodes
limit.

## Exploration

Exploration in continous actions space is very different from the epsilon-greedy approach used in discrete action 
spaces. Exploration is added by adding  random noise to the selected actions after the neural (actor) network has done 
it computations. The original DDPG paper uses the Ornstein-Uhlenbeck process for adding noise wich add a few more 
hyperparameters to worry about: theta and sigma.

In order to reduce the exploration as training goes by, a noise scaling procedure is implemented. The noise scaling
is reduced from  1.0 to 0.1 over a number of episodes.
I tried both with OU noise and normal distribution where normal distribution seems to lead to faster training.

## Best configuration run

The configuration were generated using the gen_params.py tool using the config template file template.json file as input.
Paramemters and values you want check out is entered here as lists and the output can be used in the train.py which will runn
all (or some) of configurations and place the resutls in the output folder. Try --help on these tools. Default values 
are found in maddpg_agent.py

    {
        "n_episodes": 2500,
        "seeds": 1,
        "learn_every": 5,
        "hidden_units": 128,
        "lr_critic": 0.001,
        "lr_actor": 0.0001,
        "weight_deacy": 1,
        "learn_num": 5,
        "gamma": 0.99,
        "tau": 0.07,
        "ou_sigma": 0.3,
        "ou_theta": 0.15,
        "eps_start": 1.0,
        "eps_ep_end": 400,
        "eps_final": 0.1,
        "batch_size": 128
  }

![Alt text](results/plot_3.png?raw=true "Training results best configuration")

# Improvements

 * Training on critic per agent is not very efficient. Creating a centrally trained critic will probably
save resources and training time. I did a effort on this, but the implementation lost its cleanliness of using separate 
DDPG and I lost the track.

 * Use Prioritized Experience Replay to better utilize the replay buffer with good samples (that has effect on training).

 * There are other methods than policy gradient that could be used for solving this problem. One is Multi Agent Proximal 
Policy Optimization (MAPPO) and A3C.
import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256       # minibatch size : 128
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 5           # number of learning passes
GAMMA = 0.99            # discount factor
TAU = 8e-3              # for soft update of target parameters
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay

EPISODES_BEFORE_TRAINING = 100
NUM_LEARN_STEPS_PER_ENV_STEP = 3


DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agents = []

class MADDPG_Central_Critic_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed, config={}):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        self.buffer_size = config.get("buffer_size", BUFFER_SIZE)
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.lr_actor = config.get("lr_actor", LR_ACTOR)
        self.lr_critic = config.get("lr_critic", LR_CRITIC)
        self.weight_decay = config.get("weight_decay", WEIGHT_DECAY)
        self.learn_every = config.get("learn_every", LEARN_EVERY)
        self.learn_num = config.get("learn_num", LEARN_NUM)
        self.gamma = config.get("gamma", GAMMA)
        self.tau = config.get("tau", TAU)
        self.hidden_units = config.get("hidden_units", 256)
        self.ou_sigma = config.get("ou_sigma", OU_SIGMA)
        self.ou_theta = config.get("ou_theta", OU_THETA)
        self.eps_start = config.get("eps_start", EPS_START)
        self.eps_ep_end = config.get("eps_end", EPS_EP_END)
        self.eps_final = config.get("eps_end", EPS_FINAL)
        self.batch_norm = False

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.eps = self.eps_start
        self.eps_decay = 1/(self.eps_ep_end*self.learn_num)  # set decay rate based on epsilon end target
        self.timestep = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, use_batch_norm=self.batch_norm,
                                 fc1_units=self.hidden_units,
                                 fc2_units=self.hidden_units).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed ,  use_batch_norm=self.batch_norm,
                                  fc1_units=self.hidden_units, fc2_units=self.hidden_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, use_batch_norm=self.batch_norm,
                                   fc1_units=self.hidden_units,
                                   fc2_units=self.hidden_units).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, use_batch_norm=self.batch_norm,
                                    fc1_units=self.hidden_units,
                                    fc2_units=self.hidden_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay)

        # Noise process
        #self.noise = OUNoise((1, action_size), random_seed)
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, random_seed)

    def step(self, state, action, reward, next_state, done, agent_number):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.timestep += 1
        full_states = np.reshape(state, newshape=(-1))
        full_next_states = np.reshape(next_state, newshape=(-1))

        # Save experience / reward
        #print("to memory", state, action, reward, next_state, done)
        self.memory.add(full_states, state, action, reward, full_next_states, next_state, done)
        # Learn, if enough samples are available in memory and at learning interval settings
        if len(self.memory) > self.batch_size and self.timestep % self.learn_every == 0:
                for _ in range(self.learn_num):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma, agent_number)

    def act(self, states, add_noise):
        """Returns actions for both agents as per current policy, given their respective states."""
        print(states.shape)
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        print("in act()", states.size(), actions, self.num_agents, self.action_size)
        self.actor_local.eval()
        with torch.no_grad():
            # get action for each agent and concatenate them
            #for agent_num, state in enumerate(states):
                #print("hei", agent_num, state, states)
                actions = self.actor_local(states).cpu().data.numpy()
                #print("action", agent_num, action)
                #actions[agent_num, :] = action
        self.actor_local.train()
        # add noise to actions

        if add_noise:
            print(type(actions),actions, actions.shape, self.eps, self.noise.sample())
        #    actions += self.eps * self.noise.sample()
        actions = np.clip(actions, -1, 1)
        print("act() return actions", actions, actions.shape)
        return actions

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_number):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        full_states, states, actions, rewards, full_next_states, next_states, dones = experiences
        print("experiences", states.size(), actions.size(), rewards.size(), next_states.size(), dones.size())

        critic_full_next_actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=DEVICE)
        for agent_id, agent in enumerate(agents):
            agent_next_state = next_states[:, agent_id, :]
            critic_full_next_actions[:, agent_id, :] = agent.actor_target.forward(agent_next_state)
        critic_full_next_actions = critic_full_next_actions.view(-1, 2 * self.action_size)

        print(critic_full_next_actions)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        #actions_next = self.actor_target(next_states[agent_number])
        actions_next = self.actor_target(next_states[agent_number])
        print("next_states", next_states[agent_number])
        print("learning actions_next", actions_next)
        print("learning actions", actions)
        print("learning dones", dones)
        # Construct next actions vector relative to the agent
        print("sized", actions.size(), actions_next.size())
        #if agent_number == 0:
        #    actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        #else:
        #    actions_next = torch.cat((actions[:,:2], actions_next), dim=1)

        # Compute Q targets for current states (y_i)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states[agent_number])
        # Construct action prediction vector relative to each agent
        if agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
        # Compute actor loss
        actor_loss = -self.critic_local(states[agent_number], actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        # update noise decay parameter
        self.eps -= self.eps_decay
        self.eps = max(self.eps, EPS_FINAL)
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


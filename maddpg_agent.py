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
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 5           # number of learning passes
GAMMA = 0.99            # discount factor
TAU = 8e-3              # for soft update of target parameters
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 1.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0.1           # final value for epsilon after decay

EPISODES_BEFORE_TRAINING = 100
NOISE_REDUCTION=0.999


DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG(object):
    '''The main class that defines and strains all the agents'''

    def __init__(self, state_size, action_size, num_agents, config={}):

        self.config(config)
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.whole_action_dim = self.action_size * self.num_agents
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)  # Replay memory
        self.maddpg_agents = [DDPG(state_size, action_size, num_agents),
                              DDPG(state_size, action_size, num_agents)]  # create agents



    def config(self, config):
        self.buffer_size = config.get("buffer_size", BUFFER_SIZE)
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.lr_actor = config.get("lr_actor", LR_ACTOR)
        self.lr_critic = config.get("lr_critic", LR_CRITIC)
        self.weight_decay = config.get("weight_decay", WEIGHT_DECAY)
        #self.learn_every = config.get("learn_every", LEARN_EVERY)
        self.learn_num = config.get("learn_num", LEARN_NUM)
        self.eps_ep_end = config.get("eps_ep_end", EPS_EP_END)
        self.episodes_before_training = config.get("episodes_before_training", EPISODES_BEFORE_TRAINING)
        self.gamma = config.get("gamma", GAMMA)


    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()

    def step(self, i_episode, states, actions, rewards, next_states, dones):
        # for stepping maddpg
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # index 0 is for agent 0 and index 1 is for agent 1
        #print("step next_states", next_states[0])
        full_states = np.reshape(states, newshape=(-1))
        full_next_states = np.reshape(next_states, newshape=(-1))

        # Save experience / reward
        self.memory.add(full_states, states, actions, rewards, full_next_states, next_states, dones)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and i_episode > self.episodes_before_training:
            for _ in range(self.learn_num):  # learn multiple times at every step
                for agent_no in range(self.num_agents):
                    samples = self.memory.sample()
                    self.learn(samples, agent_no, self.gamma)
                self.soft_update_all()

    def soft_update_all(self):
        # soft update all the agents
        for agent in self.maddpg_agents:
            agent.soft_update_all()

    def learn(self, samples, agent_no, gamma):
        # for learning MADDPG
        full_states, states, actions, rewards, full_next_states, next_states, dones = samples

        critic_full_next_actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=DEVICE)
        # print("critic_full_next_actions", critic_full_next_actions.size())
        # print ("full_states",full_states.size())
        # print("states",states.size())
        # print("actions",actions.size())
        # print("rewards", rewards.size())
        # print("full_next_states",full_next_states.size())
        # print("next_states", next_states.size())
        # print("dones", dones.size())

        for agent_id, agent in enumerate(self.maddpg_agents):
            agent_next_state = next_states[:, agent_id, :]
            critic_full_next_actions[:, agent_id, :] = agent.actor_target.forward(agent_next_state)
        critic_full_next_actions = critic_full_next_actions.view(-1, self.whole_action_dim)

        agent = self.maddpg_agents[agent_no]
        agent_state = states[:, agent_no, :]
        actor_full_actions = actions.clone()  # create a deep copy
        actor_full_actions[:, agent_no, :] = agent.actor_local.forward(agent_state)
        actor_full_actions = actor_full_actions.view(-1, self.whole_action_dim)

        full_actions = actions.view(-1, self.whole_action_dim)

        agent_rewards = rewards[:, agent_no].view(-1, 1)  # gives wrong result without doing this
        agent_dones = dones[:, agent_no].view(-1, 1)  # gives wrong result without doing this
        experiences = (full_states, actor_full_actions, full_actions, agent_rewards, \
                       agent_dones, full_next_states, critic_full_next_actions)
        agent.learn(experiences, gamma)


    def act(self, full_states, i_episode, add_noise=True):
        # all actions between -1 and 1
        actions = []
        for agent_id, agent in enumerate(self.maddpg_agents):
            action = agent.act(np.reshape(full_states[agent_id, :], newshape=(1, -1)), i_episode, add_noise)
            action = np.reshape(action, newshape=(1, -1))
            actions.append(action)
        actions = np.concatenate(actions, axis=0)
        return actions

    def save_maddpg(self):
        for agent_id, agent in enumerate(self.maddpg_agents):
            torch.save(agent.actor_local.state_dict(), 'models/checkpoint_actor_local_' + str(agent_id) + '.pth')
            torch.save(agent.critic_local.state_dict(), 'models/checkpoint_critic_local_' + str(agent_id) + '.pth')

    def load_maddpg(self):
        for agent_id, agent in enumerate(self.maddpg_agents):
            # Since the model is trained on gpu, need to load all gpu tensors to cpu:
            agent.actor_local.load_state_dict(torch.load('models/checkpoint_actor_local_' + str(agent_id) + '.pth',
                                                         map_location=lambda storage, loc: storage))
            agent.critic_local.load_state_dict(torch.load('models/checkpoint_critic_local_' + str(agent_id) + '.pth',
                                                          map_location=lambda storage, loc: storage))

            agent.noise_scale = self.eps_ep_end  # initialize to the final epsilon value upon training


class DDPG(object):
    """Interacts with and learns from the environment.
    There are two agents and the observations of each agent has 24 dimensions. Each agent's action has 2 dimensions.
    Will use two separate actor networks (one for each agent using each agent's observations only and output that agent's action).
    The critic for each agents gets to see the actions and observations of all agents. """

    def __init__(self, state_size, action_size, num_agents, config = {}):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state for each agent
            action_size (int): dimension of each action for each agent
        """
        self.state_size = state_size
        self.seed = config.get("seed", 1)
        self.batch_norm = False
        self.action_size = action_size
        self.batch_norm = False

        self.config(config)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, self.seed, use_batch_norm=self.batch_norm,
                                 fc1_units=self.hidden_units,
                                 fc2_units=self.hidden_units).to(device)
        self.actor_target = Actor(state_size, action_size, self.seed ,  use_batch_norm=self.batch_norm,
                                  fc1_units=self.hidden_units,
                                  fc2_units=self.hidden_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * num_agents, action_size * num_agents, self.seed, use_batch_norm=self.batch_norm,
                                   fc1_units=self.hidden_units,
                                   fc2_units=self.hidden_units).to(device)
        self.critic_target = Critic(state_size * num_agents, action_size * num_agents, self.seed, use_batch_norm=self.batch_norm,
                                    fc1_units=self.hidden_units,
                                    fc2_units=self.hidden_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)


        # Noise process
        self.noise = OUNoise(action_size, self.seed)  # single agent only
        self.noise_scale = self.eps_start
        # Make sure target is initialized with the same weight as the source (makes a big difference)
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

    def config(self, config):
        self.lr_actor = config.get("lr_actor", 1e-3)
        self.lr_critic =config.get("lr_critic", 3e-3)
        self.weight_decay = config.get("weight_decay", WEIGHT_DECAY)
        self.tau = config.get("tau", TAU)
        self.hidden_units = config.get("hidden_units", 128)
        self.ou_sigma = config.get("ou_sigma", OU_SIGMA)
        self.ou_theta = config.get("ou_theta", OU_THETA)
        self.eps_start = config.get("eps_start", EPS_START) # noise scaling start value
        self.eps_final = config.get("eps_end", EPS_FINAL)  # final noise scale value
        self.eps_ep_end = config.get("eps_ep_end", EPS_EP_END)
        self.episodes_before_training = config.get("episodes_before_training", EPISODES_BEFORE_TRAINING)


    def act(self, states, i_episode, add_noise=True):
        """Returns actions for given state as per current policy."""

        if i_episode > self.episodes_before_training and self.noise_scale > self.eps_final:
            # self.noise_scale *= NOISE_REDUCTION
            self.noise_scale = NOISE_REDUCTION ** (i_episode - self.episodes_before_training)
        # else keep the previous value

        if not add_noise:
            self.noise_scale = 0.0

        states = torch.from_numpy(states).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()

        # add noise
        actions += self.noise_scale * self.add_noise2()  # uniform
        #actions += self.noise_scale*self.noise.sample() # ou

        return np.clip(actions, -1, 1)

    def learn(self, experiences, gamma):
        # for MADDPG
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
        full_states, actor_full_actions, full_actions, agent_rewards, agent_dones, full_next_states, critic_full_next_actions = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get Q values from target models
        Q_target_next = self.critic_target(full_next_states, critic_full_next_actions)
        # Compute Q targets for current states (y_i)
        Q_target = agent_rewards + gamma * Q_target_next * (1 - agent_dones)
        # Compute critic loss
        Q_expected = self.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(input=Q_expected,
                                 target=Q_target)  # target=Q_targets.detach() #not necessary to detach
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1.0) #clip the gradient for the critic network (Udacity hint)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actor_loss = -self.critic_local.forward(full_states,
                                                actor_full_actions).mean()  # -ve b'cse we want to do gradient ascent
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update_all(self):
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def add_noise2(self):
        noise = 0.3 * np.random.randn(1,
                                      self.action_size)  # sigma of 0.5 as sigma of 1 will have alot of actions just clipped
        return noise

    def reset(self):
        pass


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process.
        Params
        ======
            mu (float)    : long-running mean
            theta (float) : speed of mean reversion
            sigma (float) : volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    # actions += self.noise.sample()
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["full_state", "state", "action", "reward", \
                                                                "full_next_state", "next_state", "done"])

    def add(self, full_state, state, action, reward, full_next_state, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(full_state, state, action, reward, full_next_state, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        full_states = torch.from_numpy(np.array([e.full_state for e in experiences if e is not None])).float().to(
            DEVICE)
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        full_next_states = torch.from_numpy(
            np.array([e.full_next_state for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(
            DEVICE)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            DEVICE)

        return (full_states, states, actions, rewards, full_next_states, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


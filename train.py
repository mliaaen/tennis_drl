# main function that sets up environments
# and performs training loop

from collections import deque
from maddpg_agent import MADDPG
import numpy as np
import os
import time
import torch
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import json

import csv
import copy
import time
import argparse


# some globals we dont want to change
N_EPISODES = 2500
SOLVED_SCORE = 0.5
CONSEC_EPISODES = 100
PRINT_EVERY = 10
ADD_NOISE = True



class MADDPG_Runner():
    
    def __init__(self, config):
        """

        :rtype: object
        """
        super(MADDPG_Runner, self).__init__()

        self.agents = []

        self.env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
        #self.env = UnityEnvironment(file_name="Tennis.app")

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # reset the environment
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        # number of agents
        self.num_agents = len(env_info.agents)
        print('Number of agents:', self.num_agents)

        # size of each action
        self.action_size = self.brain.vector_action_space_size
        print('Size of each action:', self.action_size)
        # examine the state space
        states = env_info.vector_observations
        self.state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], self.state_size))
        print('The state for the first agent looks like: \n{}\n'.format(states[0]))

        self.config(config)

    def config(self, config):
        print("Prepare for new configuration. Make new MADDPG agent.")
        self.n_episodes = config.get("n_episodes", N_EPISODES)
        self.solved_score = config.get("solved_score", SOLVED_SCORE)
        self.conseq_episodes = config.get("conseq_episodes", CONSEC_EPISODES)
        self.seed = config.get("seed", 1)
        self.MADDPG_obj = MADDPG(state_size=self.state_size, action_size=self.action_size, num_agents=self.num_agents, config=config)

    def seeding(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def reset_agents(self):
        for agent in self.agents:
            agent.reset()

    def learning_step(self, states, actions, rewards, next_states, done):
        print("learning step", states, next_states, rewards, done, actions)
        for i, agent in enumerate(self.agents):
            agent.step(states, actions, rewards, next_states, done, i)

    ## Training loop


    def training_loop(self,  t_max = 1000, stop_when_done=True):

        # initialize scoring
        scores_window = deque(maxlen=CONSEC_EPISODES)
        moving_average = []
        scores_all = []
        best_score = -np.inf
        best_episode = 0
        already_solved = False
        self.seeding()

        scores_deque = deque(maxlen=100)
        scores_list = []
        scores_list_100_avg = []

        for i_episode in range(1, self.n_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            states = env_info.vector_observations  # get the current states (for all agents)
            scores = np.zeros(self.num_agents)  # initialize the score (for each agent in MADDPG)
            num_steps = 0
            actions = []
            for _ in range(t_max):
                actions = self.MADDPG_obj.act(states, i_episode, add_noise=ADD_NOISE)
                env_info = self.env.step(actions)[self.brain_name]  # send all actions to the environment
                next_states = env_info.vector_observations  # get next state (for each agent in MADDPG)
                rewards = env_info.rewards  # get rewards (for each agent in MADDPG)
                dones = env_info.local_done  # see if episode finished
                scores += rewards  # update the score (for each agent in MADDPG)
                self.MADDPG_obj.step(i_episode, states, actions, rewards, next_states,
                                dones)  # train the MADDPG_obj
                states = next_states  # roll over states to next time step
                num_steps += 1
                if np.any(dones):  # exit loop if episode finished
                    break
                # print('Total score (averaged over agents) this episode: {}'.format(np.mean(score)))

            scores_deque.append(np.max(scores))
            scores_list.append(np.max(scores))
            scores_list_100_avg.append(np.mean(scores_deque))

            # print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {}'.format(i_episode, np.mean(scores_deque), score), end="")
            if i_episode % PRINT_EVERY == 0:
                print('Episode {}\tAverage Score: {:.2f}\tCurrent Score: {}'.format(i_episode, np.mean(scores_deque),
                                                                                np.max(scores)))
                print('Noise Scaling: {}, Memory size: {} and Num Steps: {}'.format(self.MADDPG_obj.maddpg_agents[0].noise_scale,
                                                                                len(self.MADDPG_obj.memory), num_steps))
                print("last 10", scores_list[-10:])
                print("last actions", actions)

            if i_episode % 500 == 0:
                self.MADDPG_obj.save_maddpg()
                print('Saved Model: Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

            if np.mean(scores_deque) > self.solved_score and len(scores_deque) >= 100:
                self.MADDPG_obj.save_maddpg()
                print('Goal reached. Saved Model: Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                if stop_when_done:
                    break
            

        return scores_list, scores_list_100_avg, i_episode



def plot_scores(scores, scores_avg, annotation, output=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_list) + 1), scores_list)
    plt.plot(np.arange(1, len(scores_list_100_avg) + 1), scores_list_100_avg)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    fig.savefig(os.path.join(output,'plot_{}.png'.format(annotation)))


#
def train(configs, output, ids):

    print("Nmber of configs: %d"%len(configs))

    scores = []

    with open(os.path.join(output,'scan_report.csv'), 'w', newline='') as csvfile:

        fieldnames = list(configs[0].keys())
        fieldnames = fieldnames + ["best_score", "avg_score", "solved", "time"]
        print(fieldnames, ids)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        runner = MADDPG_Runner(configs[0])
        for cnf_no, config in enumerate(configs):

            if not ids and str(cnf_no) not in ids:
                print("skipping", cnf_no)
                continue

            print("Now running with config:", config)
            start_time = time.time()
            runner.config(config)
            scores_list, scores_list_100_avg, episode = runner.training_loop()
            time_passed = time.time() - start_time
            best_score = max(scores_list)
            avg_100_score = max(scores_list_100_avg)
            scores.append(best_score)
            report_line = copy.copy(config)
            report_line["best_score"] = max(scores_list)
            report_line["avg_score"] = max(scores_list_100_avg)
            report_line["solved"] = episode
            report_line["time"] = time_passed
            writer.writerow(report_line)
            with open(os.path.join(output, "scores_file_{}.json".format(cnf_no)), "w") as write_file:
                json.dump((scores_list, scores_list_100_avg), write_file)

            plot_scores(scores_list, scores_list_100_avg, annotation=cnf_no)
            time.sleep(4)
        print (scores)




if __name__ == '__main__':

    description = """
Running the training loop
"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', action="store", dest="configfile", default="The json formatted configs file (see gen_configs.py)")
    parser.add_argument('--output', action="store", dest="output_dir", default="", help="Folder where the results are stored")
    parser.add_argument('--ids', action="append", dest="ids", default="[]", help="Run selected config, default is all")

    args = parser.parse_args()
    print(args)
    with open(args.configfile,"r") as f:
        configs = json.load(f)
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        train (configs, args.output_dir, args.ids)
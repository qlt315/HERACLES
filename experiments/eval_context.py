import math

import torch
import numpy as np
import gym
from envs.env_proposed_origin import EnvProposed_origin
from envs.env_proposed_erf import EnvProposed_erf
from envs.env_tem import EnvTEM
from envs.env_sse import EnvSSE
from torch.utils.tensorboard import SummaryWriter
from baselines import amac
from rainbow_dqn.rainbow_replay_buffer import *
from rainbow_dqn.rainbow_agent import DQN
import argparse
import random
import tools.saving_loading as sl
import time
import pandas as pd
import os
from experiments.eval_bad_actions import get_top_k_values
from collections import Counter
from scipy.io import savemat
from decimal import Decimal, getcontext



class Runner:
    def __init__(self, args, env, number, seed):
        self.args = args
        self.number = number
        self.seed = seed
        self.env = env
        # self.env = EnvSSE()
        # self.env = EnvTEM()
        # self.env.seed(seed)
        # self.env.action_space.seed(seed)
        # self.env_evaluate.seed(seed)
        # self.env_evaluate.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.n
        self.args.limit = self.env.slot_num  # Maximum number of steps per episode
        print("env name:", self.env.name)
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("limit={}".format(self.args.limit))

        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)

        self.algorithm = 'dqn'
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'rainbow_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += '_double'
            if args.use_dueling:
                self.algorithm += '_dueling'
            if args.use_noisy:
                self.algorithm += '_noisy'
            if args.use_per:
                self.algorithm += '_per'
            if args.use_n_steps:
                self.algorithm += "_n_steps"

        # self.writer = SummaryWriter(
        #     log_dir='runs/dqn/{}_env_{}_number_{}_seed_{}'.format(self.algorithm, scheme_name, number, seed))

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
        if args.use_noisy:  # If choose to use Noisy net, then the epsilon is no looger needed
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def run(self, ):
        # self.evaluate_policy()
        while self.total_steps < self.args.max_train_steps:
            state = self.env.reset()
            done = False
            steps = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=self.epsilon)
                next_state, reward, done = self.env.step(action)
                self.total_steps += 1

                if not self.args.use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                # When dead or win or reaching the max_steps, done will be Ture, we need to distinguish them;
                # terminal means dead or win,there is no next state c';
                # but when reaching the max_steps,there is a next state c' actually.
                if done and steps != self.args.limit:
                    terminal = True
                else:
                    terminal = False

                self.replay_buffer.store_transition(state, action, reward, next_state, terminal,
                                                    done)  # Store the transition
                state = next_state

                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)




def run_all_table(seed):
    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    show_action_num = 3

    context_list = ["sunny", "snow", "fog", "motorway", "night", "rain", "mix"]
    algorithms = ["Proposed_erf", "Proposed_origin", "SSE", "TEM", "DQN", "AMAC"]
    columns = ["Delay", "Energy", "Accuracy", "Accuracy Vio Rate", "Accuracy Deviation", "Re-trans Num", "Reward",
               "Most Picked " + str(show_action_num) + " Action"]

    table_data = np.empty([len(context_list), len(algorithms), len(columns)], dtype='U100')

    drl_algorithm_list = ["rainbow_dqn", "dqn"]
    # drl_algorithm_list = []
    env_list = [EnvProposed_erf(), EnvProposed_origin(), EnvSSE(), EnvTEM()]
    env_num = len(set(type(obj) for obj in env_list))

    def create_excel(table_data, file_name):
        # Assume data is a 3D array with shape (6 contexts, 6 algorithms, 7 metrics)
        # 6 contexts: ["snow", "fog", "motorway", "night", "rain", "sunny"]
        # 6 algorithms: each context includes 6 rows of data
        # 7 metrics: ["Reward", "Delay", "Energy", "Accuracy", "Accuracy", "Re-trans", "Actions"]

        # Create an empty DataFrame
        df = pd.DataFrame(columns=["Context", "Algorithm"] + columns)

        # Fill in the data
        for i, context in enumerate(context_list):
            for j, algo in enumerate(algorithms):
                # Get the corresponding data values, assuming data has shape (6, 6, 7)
                row_data = table_data[i, j, :]
                # Create the row
                row = [context, algo] + list(row_data)
                # Add the row to the DataFrame
                df.loc[len(df)] = row

        # Output to an Excel file
        df.to_excel(file_name, index=False)

    length = 3000  # Number of steps / episode
    number = 1  # Number of episode to train
    steps = number * length  # Total step number
    for w in range(len(context_list)):
        scheme_id = 0
        for dql_algo_id in range(len(drl_algorithm_list)):
            algorithm = drl_algorithm_list[dql_algo_id]
            parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
            parser.add_argument("--max_train_steps", type=int, default=int(steps), help=" Maximum number of training steps")
            parser.add_argument("--evaluate_freq", type=float, default=1e3,
                                help="Evaluate the policy every 'evaluate_freq' steps")
            parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

            parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
            parser.add_argument("--batch_size", type=int, default=256, help="batch size")
            parser.add_argument("--hidden_dim", type=int, default=256,
                                help="The number of neurons in hidden layers of the neural network")
            parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
            parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
            parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
            parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
            parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5),
                                help="How many steps before the epsilon decays to the minimum")
            parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
            parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
            parser.add_argument("--target_update_freq", type=int, default=200,
                                help="Update frequency of the target network(hard update)")
            parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
            parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
            parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
            parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
            parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")
            if algorithm == "rainbow_dqn":
                parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
                parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
                parser.add_argument("--use_noisy", type=bool, default=True, help="Whether to use noisy network")
                parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
                parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")
            else:
                parser.add_argument("--use_double", type=bool, default=False, help="Whether to use double Q-learning")
                parser.add_argument("--use_dueling", type=bool, default=False, help="Whether to use dueling network")
                parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
                parser.add_argument("--use_per", type=bool, default=False, help="Whether to use PER")
                parser.add_argument("--use_n_steps", type=bool, default=False, help="Whether to use n_steps Q-learning")
            args = parser.parse_args()

            for env_id in range(env_num):
                env_index = 0
                env = env_list[env_id]
                if algorithm == "dqn" and env.name != "proposed_erf":
                    continue
                runner = Runner(args=args, env=env, number=1, seed=seed)
                folder_path = context_list[w] + "_models"
                if context_list[w]=="mix":
                    runner.env.context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
                else:
                    runner.env.context_list = [context_list[w]] * (len(context_list)-1)
                # load the model
                # runner.agent.net, runner.agent.target_net = sl.load_nn_model_diff_context(runner, folder_path)
                runner.agent.net, runner.agent.target_net = sl.load_nn_model(runner)
                print("env name:", env.name)
                print("current context:", context_list[w])
                print("algorithm:", runner.algorithm)
                print("Scheme ID:",scheme_id)
                runner.run()
                # save the data
                delay_mean = round(np.mean(runner.env.total_delay_list),2)
                # delay_std  = round(np.std(runner.env.total_delay_list),2)
                # delay_str = str(delay_mean) + "\u00B1" + str(delay_std)
                delay_var = round(np.var(runner.env.total_delay_list),4)
                delay_str = str(delay_mean) + "("  + str(delay_var) + ")"
                table_data[w,scheme_id,0] = delay_str


                energy_mean = round(np.mean(runner.env.total_energy_list),2)
                # energy_std  = round(np.std(runner.env.total_energy_list),2)
                # energy_str = str(energy_mean) + "\u00B1" + str(energy_std)
                energy_var = round(np.var(runner.env.total_energy_list), 4)
                energy_str = str(energy_mean) + "(" + str(energy_var) + ")"
                table_data[w,scheme_id,1] = energy_str

                acc_mean = round(np.mean(runner.env.acc_exp_list),2)
                # acc_std  = round(np.std(runner.env.acc_exp_list),2)
                # acc_str = str(acc_mean) + "\u00B1" + str(acc_std)
                acc_var = round(np.var(runner.env.acc_exp_list), 4)
                acc_str = str(acc_mean) + "(" + str(acc_var) + ")"
                table_data[w,scheme_id,2] = acc_str

                acc_vio_rate_mean = round(np.mean(runner.env.episode_acc_vio_num_list),5)
                # acc_vio_std = np.std(runner.env.episode_acc_vio_num_list)
                acc_vio_rate_str = str(acc_vio_rate_mean)
                table_data[w, scheme_id, 3] = acc_vio_rate_str

                acc_vio_mean = round(np.mean(runner.env.episode_acc_vio_list),2)
                # acc_vio_std = np.std(runner.env.episode_acc_vio_num_list)
                acc_vio_str = str(acc_vio_mean)
                table_data[w, scheme_id, 4] = acc_vio_rate_str


                re_trans_mean = int(np.mean(runner.env.re_trans_list))
                # re_trans_std = np.std(runner.env.episode_re_trans_num_list)
                re_trans_var = int(np.var(runner.env.re_trans_list))

                re_trans_str = str(re_trans_mean) + "(" + str(re_trans_var) + ")"
                table_data[w, scheme_id, 5] = re_trans_mean

                reward_mean = round(np.mean(runner.env.reward_list),2)
                # reward_std = round(np.std(runner.env.reward_list),2)
                # reward_str = str(reward_mean) + "\u00B1" + str(reward_std)
                reward_var = round(np.var(runner.env.reward_list), 4)
                reward_str = str(reward_mean) + "(" + str(reward_var) + ")"
                table_data[w, scheme_id, 6] = reward_str


                action_str = ""
                index, values = most_picked_action = get_top_k_values(runner.env.action_freq_list, show_action_num)
                for act in range(show_action_num-1, -1, -1):
                    action_index = index[act]
                    action_freq = round(values[act] / runner.env.slot_num * 100, 2)
                    if action_str == "":
                        action_str = runner.env.get_action_name(action_index) + "(" + str(action_freq) + "%)"
                    else:
                        action_str = action_str + ";" + runner.env.get_action_name(action_index) + "(" + str(action_freq) + "%)"
                # print(action_str)
                table_data[w, scheme_id, 7] = action_str
                scheme_id = scheme_id + 1
                runner.env.reset()
        # amac evaluation
        print("Evaluating AMAC")  # amac
        action_str = "[1,2,3,4]"
        runner = amac.Amac(is_test=True)
        if context_list[w] == "mix":
            runner.context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
        else:
            runner.context_list = [context_list[w]] * (len(context_list) - 1)
        runner.run()
        delay_mean = round(np.mean(runner.total_delay_list),2)
        # delay_std = round(np.std(runner.total_delay_list),2)
        # delay_str = str(delay_mean) + "\u00B1" + str(delay_std)
        delay_var = round(np.var(runner.total_delay_list), 4)
        delay_str = str(delay_mean) + "(" + str(delay_var) + ")"
        table_data[w, scheme_id, 0] = delay_str

        energy_mean = round(np.mean(runner.total_energy_list),2)
        # energy_std = round(np.std(runner.total_energy_list),2)
        # energy_str = str(energy_mean) + "\u00B1" + str(energy_std)
        energy_var = round(np.var(runner.total_energy_list), 4)
        energy_str = str(energy_mean) + "(" + str(energy_var) + ")"
        table_data[w, scheme_id, 1] = energy_str

        acc_mean = round(np.mean(runner.total_acc_list),2)
        # acc_std = round(np.std(runner.total_acc_list),2)
        # acc_str = str(acc_mean) + "\u00B1" + str(acc_std)
        acc_var = round(np.var(runner.total_acc_list), 4)
        acc_str = str(acc_mean) + "(" + str(acc_var) + ")"
        table_data[w, scheme_id, 2] = acc_str

        acc_vio_rate_str = str(int(runner.acc_vio_num))
        table_data[w, scheme_id, 3] = acc_vio_rate_str

        acc_vio_mean = round(np.mean(runner.episode_acc_vio_list), 2)
        acc_vio_str = str(acc_vio_mean)
        table_data[w, scheme_id, 4] = acc_vio_rate_str

        re_trans_mean = int(np.mean(runner.re_trans_num_list))
        re_trans_var = int(np.var(runner.re_trans_num_list))
        re_trans_str = str(re_trans_mean) + "(" + str(re_trans_var) + ")"
        table_data[w, scheme_id, 5] = re_trans_mean

        reward_mean = round(np.mean(runner.step_reward_list),2)
        # reward_std = round(np.std(runner.step_reward_list),2)
        # reward_str = str(reward_mean) + "\u00B1" + str(reward_std)
        reward_var = round(np.var(runner.step_reward_list), 4)
        reward_str = str(reward_mean) + "(" + str(reward_var) + ")"
        table_data[w, scheme_id, 6] = reward_str

        counter = Counter(runner.action_list)
        most_common_elements = counter.most_common(show_action_num)
        for act, freq in most_common_elements:
            action_freq = round(freq / runner.slot_num * 100, 2)
            if action_str == "[1,2,3,4]+":
                action_str = action_str + act + "(" + str(action_freq) + "%)"
            else:
                action_str = action_str + ";" + act + "(" + str(action_freq) + "%)"
        table_data[w, scheme_id, 7] = action_str


    # output the table as an Excel file
    mat_name = "experiments/diff_context_data/performance_table_diff_context.mat"
    savemat(mat_name,{"performance_table_diff_context": table_data})
    create_excel(table_data, "experiments/diff_context_data/performance_table_diff_context.xlsx")


def run_single_table(seed, scheme_name):
    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    show_action_num = 3

    context_list = ["sunny", "snow", "fog", "motorway", "night", "rain", "mix"]
    algorithms = ["Proposed_erf", "Proposed_origin", "SSE", "TEM", "DQN", "AMAC"]
    columns = ["Delay", "Energy", "Accuracy", "Accuracy Vio Rate", "Accuracy Deviation", "Re-trans Num", "Reward",
               "Most Picked " + str(show_action_num) + " Action"]

    table_data = np.empty([len(context_list), len(algorithms), len(columns)], dtype='U100')
    env_list = []
    drl_algorithm_list = []
    if scheme_name == "proposed_erf":
        env_list = [EnvProposed_erf()]
        drl_algorithm_list = ["rainbow_dqn"]
    elif scheme_name == "proposed_origin":
        env_list = [EnvProposed_origin()]
        drl_algorithm_list = ["rainbow_dqn"]
    elif scheme_name == "sse":
        env_list = [EnvSSE()]
        drl_algorithm_list = ["rainbow_dqn"]
    elif scheme_name == "dqn":
        env_list = [EnvProposed_erf()]
        drl_algorithm_list = ["dqn"]
    elif scheme_name == "tem":
        env_list = [EnvTEM()]
        drl_algorithm_list = ["rainbow_dqn"]
    env_num = len(set(type(obj) for obj in env_list))

    def create_excel(table_data, file_name):
        # Assume data is a 3D array with shape (6 contexts, 6 algorithms, 7 metrics)
        # 6 contexts: ["snow", "fog", "motorway", "night", "rain", "sunny"]
        # 6 algorithms: each context includes 6 rows of data
        # 7 metrics: ["Reward", "Delay", "Energy", "Accuracy", "Accuracy", "Re-trans", "Actions"]

        # Create an empty DataFrame
        df = pd.DataFrame(columns=["Context", "Algorithm"] + columns)

        # Fill in the data
        for i, context in enumerate(context_list):
            for j, algo in enumerate(algorithms):
                # Get the corresponding data values, assuming data has shape (6, 6, 7)
                row_data = table_data[i, j, :]
                # Create the row
                row = [context, algo] + list(row_data)
                # Add the row to the DataFrame
                df.loc[len(df)] = row

        # Output to an Excel file
        df.to_excel(file_name, index=False)

    length = 3000  # Number of steps / episode
    number = 1  # Number of episode to train
    steps = number * length  # Total step number
    for w in range(len(context_list)):
        scheme_id = 0
        for dql_algo_id in range(len(drl_algorithm_list)):
            algorithm = drl_algorithm_list[dql_algo_id]
            parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
            parser.add_argument("--max_train_steps", type=int, default=int(steps), help=" Maximum number of training steps")
            parser.add_argument("--evaluate_freq", type=float, default=1e3,
                                help="Evaluate the policy every 'evaluate_freq' steps")
            parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

            parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
            parser.add_argument("--batch_size", type=int, default=256, help="batch size")
            parser.add_argument("--hidden_dim", type=int, default=256,
                                help="The number of neurons in hidden layers of the neural network")
            parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
            parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
            parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
            parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
            parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5),
                                help="How many steps before the epsilon decays to the minimum")
            parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
            parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
            parser.add_argument("--target_update_freq", type=int, default=200,
                                help="Update frequency of the target network(hard update)")
            parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
            parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
            parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
            parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
            parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")
            if algorithm == "rainbow_dqn":
                parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
                parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
                parser.add_argument("--use_noisy", type=bool, default=True, help="Whether to use noisy network")
                parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
                parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")
            else:
                parser.add_argument("--use_double", type=bool, default=False, help="Whether to use double Q-learning")
                parser.add_argument("--use_dueling", type=bool, default=False, help="Whether to use dueling network")
                parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
                parser.add_argument("--use_per", type=bool, default=False, help="Whether to use PER")
                parser.add_argument("--use_n_steps", type=bool, default=False, help="Whether to use n_steps Q-learning")
            args = parser.parse_args()

            for env_id in range(env_num):
                env = env_list[env_id]
                runner = Runner(args=args, env=env, number=1, seed=seed)
                folder_path = context_list[w] + "_models"
                if context_list[w]=="mix":
                    runner.env.context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
                else:
                    runner.env.context_list = [context_list[w]] * (len(context_list)-1)
                # load the model
                # runner.agent.net, runner.agent.target_net = sl.load_nn_model_diff_context(runner, folder_path)
                runner.agent.net, runner.agent.target_net = sl.load_nn_model(runner)
                print("env name:", env.name)
                print("current context:", context_list[w])
                print("algorithm:", runner.algorithm)
                print("Scheme ID:",scheme_id)
                runner.run()
                # save the data
                delay_mean = round(np.mean(runner.env.total_delay_list),2)
                # delay_std  = round(np.std(runner.env.total_delay_list),2)
                # delay_str = str(delay_mean) + "\u00B1" + str(delay_std)
                delay_var = round(np.var(runner.env.total_delay_list),4)
                delay_str = str(delay_mean) + "("  + str(delay_var) + ")"
                table_data[w,scheme_id,0] = delay_str


                energy_mean = round(np.mean(runner.env.total_energy_list),2)
                # energy_std  = round(np.std(runner.env.total_energy_list),2)
                # energy_str = str(energy_mean) + "\u00B1" + str(energy_std)
                energy_var = round(np.var(runner.env.total_energy_list), 4)
                energy_str = str(energy_mean) + "(" + str(energy_var) + ")"
                table_data[w,scheme_id,1] = energy_str

                acc_mean = round(np.mean(runner.env.acc_exp_list),2)
                # acc_std  = round(np.std(runner.env.acc_exp_list),2)
                # acc_str = str(acc_mean) + "\u00B1" + str(acc_std)
                acc_var = round(np.var(runner.env.acc_exp_list), 4)
                acc_str = str(acc_mean) + "(" + str(acc_var) + ")"
                table_data[w,scheme_id,2] = acc_str

                acc_vio_rate_mean = round(np.mean(runner.env.episode_acc_vio_num_list),5)
                # acc_vio_std = np.std(runner.env.episode_acc_vio_num_list)
                acc_vio_rate_str = str(acc_vio_rate_mean)
                table_data[w, scheme_id, 3] = acc_vio_rate_str

                acc_vio_mean = round(np.mean(runner.env.episode_acc_vio_list),2)
                # acc_vio_std = np.std(runner.env.episode_acc_vio_num_list)
                acc_vio_str = str(acc_vio_mean)
                table_data[w, scheme_id, 4] = acc_vio_rate_str


                re_trans_mean = int(np.mean(runner.env.re_trans_list))
                # re_trans_std = np.std(runner.env.episode_re_trans_num_list)
                re_trans_var = int(np.var(runner.env.re_trans_list))
                re_trans_str = str(re_trans_mean) + "(" + str(re_trans_var) + ")"
                table_data[w, scheme_id, 5] = re_trans_mean

                reward_mean = round(np.mean(runner.env.reward_list),2)
                # reward_std = round(np.std(runner.env.reward_list),2)
                # reward_str = str(reward_mean) + "\u00B1" + str(reward_std)
                reward_var = round(np.var(runner.env.reward_list), 4)
                reward_str = str(reward_mean) + "(" + str(reward_var) + ")"
                table_data[w, scheme_id, 6] = reward_str


                action_str = ""
                index, values = most_picked_action = get_top_k_values(runner.env.action_freq_list, show_action_num)
                for act in range(show_action_num-1, -1, -1):
                    action_index = index[act]
                    action_freq = round(values[act] / runner.env.slot_num * 100, 2)
                    if action_str == "":
                        action_str = runner.env.get_action_name(action_index) + "(" + str(action_freq) + "%)"
                    else:
                        action_str = action_str + ";" + runner.env.get_action_name(action_index) + "(" + str(action_freq) + "%)"
                print(action_str)
                table_data[w, scheme_id, 7] = action_str
                scheme_id = scheme_id + 1
                runner.env.reset()
        # amac evaluation
        if scheme_name == "amac":
            print("Evaluating AMAC")  # amac
            action_str = "[1,2,3,4]"
            runner = amac.Amac(is_test=True)
            if context_list[w] == "mix":
                runner.context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
            else:
                runner.context_list = [context_list[w]] * (len(context_list) - 1)
            runner.run()
            delay_mean = round(np.mean(runner.total_delay_list),2)
            # delay_std = round(np.std(runner.total_delay_list),2)
            # delay_str = str(delay_mean) + "\u00B1" + str(delay_std)
            delay_var = round(np.var(runner.total_delay_list), 4)
            delay_str = str(delay_mean) + "(" + str(delay_var) + ")"
            table_data[w, scheme_id, 0] = delay_str

            energy_mean = round(np.mean(runner.total_energy_list),2)
            # energy_std = round(np.std(runner.total_energy_list),2)
            # energy_str = str(energy_mean) + "\u00B1" + str(energy_std)
            energy_var = round(np.var(runner.total_energy_list), 4)
            energy_str = str(energy_mean) + "(" + str(energy_var) + ")"
            table_data[w, scheme_id, 1] = energy_str

            acc_mean = round(np.mean(runner.total_acc_list),2)
            # acc_std = round(np.std(runner.total_acc_list),2)
            # acc_str = str(acc_mean) + "\u00B1" + str(acc_std)
            acc_var = round(np.var(runner.total_acc_list), 4)
            acc_str = str(acc_mean) + "(" + str(acc_var) + ")"
            table_data[w, scheme_id, 2] = acc_str

            acc_vio_rate_str = str(int(runner.acc_vio_num))
            table_data[w, scheme_id, 3] = acc_vio_rate_str

            acc_vio_mean = round(np.mean(runner.episode_acc_vio_list), 2)
            acc_vio_str = str(acc_vio_mean)
            table_data[w, scheme_id, 4] = acc_vio_rate_str

            re_trans_mean = int(np.mean(runner.re_trans_num_list))
            re_trans_var = int(np.var(runner.re_trans_num_list))
            re_trans_str = str(re_trans_mean) + "(" + str(re_trans_var) + ")"
            table_data[w, scheme_id, 5] = re_trans_mean

            reward_mean = round(np.mean(runner.step_reward_list),2)
            # reward_std = round(np.std(runner.step_reward_list),2)
            # reward_str = str(reward_mean) + "\u00B1" + str(reward_std)
            reward_var = round(np.var(runner.step_reward_list), 4)
            reward_str = str(reward_mean) + "(" + str(reward_var) + ")"
            table_data[w, scheme_id, 6] = reward_str

            counter = Counter(runner.action_list)
            most_common_elements = counter.most_common(show_action_num)
            for act, freq in most_common_elements:
                action_freq = round(freq / runner.slot_num * 100, 2)
                if action_str == "[1,2,3,4]+":
                    action_str = action_str + act + "(" + str(action_freq) + "%)"
                else:
                    action_str = action_str + ";" + act + "(" + str(action_freq) + "%)"
            table_data[w, scheme_id, 7] = action_str
            print(action_str)

    # output the table as an Excel file
    mat_name = "experiments/diff_context_data/performance_table_diff_context.mat"
    savemat(mat_name,{"performance_table_diff_context": table_data})
    create_excel(table_data, "experiments/diff_context_data/performance_table_diff_context_" + scheme_name + ".xlsx")


def run_all(seed):
    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    # context_list = np.arange(1.5, 4.25, 0.25)
    context_list = ["sunny", "snow", "fog", "motorway", "night", "rain", "mix"]
    drl_algorithm_list = ["rainbow_dqn", "dqn"]
    env_list = [EnvProposed_erf(), EnvProposed_origin(), EnvSSE(), EnvTEM()]
    env_num = len(set(type(obj) for obj in env_list))

    rainbow_proposed_erf_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)
    rainbow_proposed_origin_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)
    rainbow_sse_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)
    rainbow_tem_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)
    amac_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)
    dqn_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)

    episode_length = 3000  # Number of steps / episode
    episode_number = 1  # Number of episode to train
    steps = episode_number * episode_length  # Total step number


    for algo_id in range(len(drl_algorithm_list)):
        algorithm = drl_algorithm_list[algo_id]
        parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
        parser.add_argument("--max_train_steps", type=int, default=int(steps), help=" Maximum number of training steps")
        parser.add_argument("--evaluate_freq", type=float, default=1e3,
                            help="Evaluate the policy every 'evaluate_freq' steps")
        parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

        parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
        parser.add_argument("--batch_size", type=int, default=256, help="batch size")
        parser.add_argument("--hidden_dim", type=int, default=256,
                            help="The number of neurons in hidden layers of the neural network")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
        parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
        parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
        parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
        parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5),
                            help="How many steps before the epsilon decays to the minimum")
        parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
        parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
        parser.add_argument("--target_update_freq", type=int, default=200,
                            help="Update frequency of the target network(hard update)")
        parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
        parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
        parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
        parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
        parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")
        if algorithm == "rainbow_dqn":
            parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
            parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
            parser.add_argument("--use_noisy", type=bool, default=True, help="Whether to use noisy network")
            parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
            parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")
        else:
            parser.add_argument("--use_double", type=bool, default=False, help="Whether to use double Q-learning")
            parser.add_argument("--use_dueling", type=bool, default=False, help="Whether to use dueling network")
            parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
            parser.add_argument("--use_per", type=bool, default=False, help="Whether to use PER")
            parser.add_argument("--use_n_steps", type=bool, default=False, help="Whether to use n_steps Q-learning")
        args = parser.parse_args()

        for c in range(len(context_list)):
            scheme_id = 0
            for env_id in range(env_num):
                if algorithm == "dqn" and env.name != "proposed_erf":
                    continue
                env_index = 0
                env = env_list[env_id]
                runner = Runner(args=args, env=env, number=1, seed=seed)
                # load the model
                # runner.agent.net, runner.agent.target_net = sl.load_nn_model(runner)
                folder_path = context_list[c] + "_models"
                runner.agent.net, runner.agent.target_net = sl.load_nn_model_diff_context(runner, folder_path)
                runner.env.context_list = [context_list[c]] * len(context_list)
                print("context:", context_list[c])
                print("algorithm:", runner.algorithm)
                runner.run()
                action_str = ""
                show_action_num = 3
                index, values = most_picked_action = get_top_k_values(runner.env.action_freq_list, show_action_num)
                for act in range(show_action_num - 1, -1, -1):
                    action_index = index[act]
                    action_freq = round(values[act] / runner.env.slot_num * 100, 2)
                    if action_str == "":
                        action_str = runner.env.get_action_name(action_index) + "(" + str(action_freq) + "%)"
                    else:
                        action_str = action_str + ";" + runner.env.get_action_name(action_index) + "(" + str(
                            action_freq) + "%)"
                print(action_str)
                # save the data
                if algorithm == "rainbow_dqn" and env.name == "proposed_origin":
                    rainbow_proposed_origin_diff_context_matrix[0, c] = runner.env.episode_total_delay_list
                    rainbow_proposed_origin_diff_context_matrix[1, c] = runner.env.total_energy_list
                    rainbow_proposed_origin_diff_context_matrix[2, c] = runner.env.acc_exp_list
                    rainbow_proposed_origin_diff_context_matrix[3, c] = runner.env.episode_acc_vio_num_list
                    rainbow_proposed_origin_diff_context_matrix[4, c] = runner.env.episode_re_trans_num_list
                    rainbow_proposed_origin_diff_context_matrix[5, c] = runner.env.reward_list
                    rainbow_proposed_origin_diff_context_matrix[6, c] = runner.env.episode_acc_vio_list

                elif algorithm == "rainbow_dqn" and env.name == "proposed_erf":
                    rainbow_proposed_erf_diff_context_matrix[0, c] = runner.env.episode_total_delay_list
                    rainbow_proposed_erf_diff_context_matrix[1, c] = runner.env.total_energy_list
                    rainbow_proposed_erf_diff_context_matrix[2, c] = runner.env.acc_exp_list
                    rainbow_proposed_erf_diff_context_matrix[3, c] = runner.env.episode_acc_vio_num_list
                    rainbow_proposed_erf_diff_context_matrix[4, c] = runner.env.episode_re_trans_num_list
                    rainbow_proposed_erf_diff_context_matrix[5, c] = runner.env.reward_list
                    rainbow_proposed_erf_diff_context_matrix[6, c] = runner.env.episode_acc_vio_list
                    aver_min_acc = np.mean(runner.env.min_acc_list[0,:])

                elif algorithm == "rainbow_dqn" and env.name == "sse":
                    rainbow_sse_diff_context_matrix[0, c] = runner.env.episode_total_delay_list
                    rainbow_sse_diff_context_matrix[1, c] = runner.env.total_energy_list
                    rainbow_sse_diff_context_matrix[2, c] = runner.env.acc_exp_list
                    rainbow_sse_diff_context_matrix[3, c] = runner.env.episode_acc_vio_num_list
                    rainbow_sse_diff_context_matrix[4, c] = runner.env.episode_re_trans_num_list
                    rainbow_sse_diff_context_matrix[5, c] = runner.env.reward_list
                    rainbow_sse_diff_context_matrix[6, c] = runner.env.episode_acc_vio_list
                elif algorithm == "rainbow_dqn" and env.name == "tem":
                    rainbow_tem_diff_context_matrix[0, c] = runner.env.episode_total_delay_list
                    rainbow_tem_diff_context_matrix[1, c] = runner.env.total_energy_list
                    rainbow_tem_diff_context_matrix[2, c] = runner.env.acc_exp_list
                    rainbow_tem_diff_context_matrix[3, c] = runner.env.episode_acc_vio_num_list
                    rainbow_tem_diff_context_matrix[4, c] = runner.env.episode_re_trans_num_list
                    rainbow_tem_diff_context_matrix[5, c] = runner.env.reward_list
                    rainbow_tem_diff_context_matrix[6, c] = runner.env.episode_acc_vio_list
                elif algorithm == "dqn":
                    dqn_diff_context_matrix[0, c] = runner.env.episode_total_delay_list
                    dqn_diff_context_matrix[1, c] = runner.env.total_energy_list
                    dqn_diff_context_matrix[2, c] = runner.env.acc_exp_list
                    dqn_diff_context_matrix[3, c] = runner.env.episode_acc_vio_num_list
                    dqn_diff_context_matrix[4, c] = runner.env.episode_re_trans_num_list
                    dqn_diff_context_matrix[5, c] = runner.env.reward_list
                    dqn_diff_context_matrix[6, c] = runner.env.episode_acc_vio_list
                runner.env.reset()

    # amac evaluation
    print("Evaluating AMAC")
    for c in range(len(context_list)):
        runner = amac.Amac(is_test=True)
        runner.context_list = [context_list[c]] * len(context_list)
        runner.run()
        action_str = "[1,2,3,4]"
        show_action_num = 3
        counter = Counter(runner.action_list)
        most_common_elements = counter.most_common(show_action_num)
        for act, freq in most_common_elements:
            action_freq = round(freq / runner.slot_num * 100, 2)
            if action_str == "[1,2,3,4]+":
                action_str = action_str + act + "(" + str(action_freq) + "%)"
            else:
                action_str = action_str + ";" + act + "(" + str(action_freq) + "%)"
        print(action_str)

        amac_diff_context_matrix[0, c] = runner.total_delay_list
        amac_diff_context_matrix[1, c] = runner.total_energy_list
        amac_diff_context_matrix[2, c] = runner.total_acc_list
        amac_diff_context_matrix[3, c] = runner.acc_vio_num
        amac_diff_context_matrix[4, c] = runner.re_trans_num_list
        amac_diff_context_matrix[5, c] = runner.step_reward_list
        amac_diff_context_matrix[6, c] = runner.episode_acc_vio_list

    # save all the data
    mat_name = "experiments/diff_context_data/diff_context_data.mat"
    savemat(mat_name,
            {"rainbow_proposed_erf_diff_context_matrix": rainbow_proposed_erf_diff_context_matrix,
             "rainbow_proposed_origin_diff_context_matrix": rainbow_proposed_origin_diff_context_matrix,
             "rainbow_sse_diff_context_matrix": rainbow_sse_diff_context_matrix,
             "rainbow_tem_diff_context_matrix": rainbow_tem_diff_context_matrix,
             "dqn_diff_context_matrix": dqn_diff_context_matrix,
             "amac_diff_context_matrix": amac_diff_context_matrix,
             "context_list":context_list,
             "aver_min_acc": aver_min_acc
             })


def run_single(seed, scheme_name):
    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    context_list = ["sunny", "snow", "fog", "motorway", "night", "rain"]

    env_list = []
    drl_algorithm_list = []
    if scheme_name == "proposed_erf":
        env_list = [EnvProposed_erf()]
        drl_algorithm_list = ["rainbow_dqn"]
    elif scheme_name == "proposed_origin":
        env_list = [EnvProposed_origin()]
        drl_algorithm_list = ["rainbow_dqn"]
    elif scheme_name == "sse":
        env_list = [EnvSSE()]
        drl_algorithm_list = ["rainbow_dqn"]
    elif scheme_name == "dqn":
        env_list = [EnvProposed_erf()]
        drl_algorithm_list = ["dqn"]
    elif scheme_name == "tem":
        env_list = [EnvTEM()]
        drl_algorithm_list = ["rainbow_dqn"]
    env_num = len(set(type(obj) for obj in env_list))

    rainbow_proposed_erf_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)
    rainbow_proposed_origin_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)
    rainbow_sse_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)
    rainbow_tem_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)
    amac_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)
    dqn_diff_context_matrix = np.zeros([7, len(context_list)], dtype=object)

    episode_length = 3000  # Number of steps / episode
    episode_number = 1  # Number of episode to train
    steps = episode_number * episode_length  # Total step number

    for algo_id in range(len(drl_algorithm_list)):
        algorithm = drl_algorithm_list[algo_id]
        parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
        parser.add_argument("--max_train_steps", type=int, default=int(steps), help=" Maximum number of training steps")
        parser.add_argument("--evaluate_freq", type=float, default=1e3,
                            help="Evaluate the policy every 'evaluate_freq' steps")
        parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

        parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
        parser.add_argument("--batch_size", type=int, default=256, help="batch size")
        parser.add_argument("--hidden_dim", type=int, default=256,
                            help="The number of neurons in hidden layers of the neural network")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
        parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
        parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
        parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
        parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5),
                            help="How many steps before the epsilon decays to the minimum")
        parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
        parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
        parser.add_argument("--target_update_freq", type=int, default=200,
                            help="Update frequency of the target network(hard update)")
        parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
        parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
        parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
        parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
        parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")
        if algorithm == "rainbow_dqn":
            parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
            parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
            parser.add_argument("--use_noisy", type=bool, default=True, help="Whether to use noisy network")
            parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
            parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")
        else:
            parser.add_argument("--use_double", type=bool, default=False, help="Whether to use double Q-learning")
            parser.add_argument("--use_dueling", type=bool, default=False, help="Whether to use dueling network")
            parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
            parser.add_argument("--use_per", type=bool, default=False, help="Whether to use PER")
            parser.add_argument("--use_n_steps", type=bool, default=False, help="Whether to use n_steps Q-learning")
        args = parser.parse_args()

        for c in range(len(context_list)):
            scheme_id = 0
            for env_id in range(env_num):
                env_index = 0
                env = env_list[env_id]
                runner = Runner(args=args, env=env, number=1, seed=seed)
                # load the model
                runner.agent.net, runner.agent.target_net = sl.load_nn_model(runner)
                runner.env.context_list = [context_list[c]] * len(context_list)
                print("context:", context_list[c])
                print("algorithm:", runner.algorithm)
                runner.run()
                action_str = ""
                show_action_num = 3
                index, values = most_picked_action = get_top_k_values(runner.env.action_freq_list, show_action_num)
                for act in range(show_action_num - 1, -1, -1):
                    action_index = index[act]
                    action_freq = round(values[act] / runner.env.slot_num * 100, 2)
                    if action_str == "":
                        action_str = runner.env.get_action_name(action_index) + "(" + str(action_freq) + "%)"
                    else:
                        action_str = action_str + ";" + runner.env.get_action_name(action_index) + "(" + str(
                            action_freq) + "%)"
                print(action_str)
                # save the data
                if algorithm == "rainbow_dqn" and env.name == "proposed_origin":
                    rainbow_proposed_origin_diff_context_matrix[0, c] = runner.env.episode_total_delay_list
                    rainbow_proposed_origin_diff_context_matrix[1, c] = runner.env.total_energy_list
                    rainbow_proposed_origin_diff_context_matrix[2, c] = runner.env.acc_exp_list
                    rainbow_proposed_origin_diff_context_matrix[3, c] = runner.env.episode_acc_vio_num_list
                    rainbow_proposed_origin_diff_context_matrix[4, c] = runner.env.episode_re_trans_num_list
                    rainbow_proposed_origin_diff_context_matrix[5, c] = runner.env.reward_list
                    rainbow_proposed_origin_diff_context_matrix[6, c] = runner.env.episode_acc_vio_list

                elif algorithm == "rainbow_dqn" and env.name == "proposed_erf":
                    rainbow_proposed_erf_diff_context_matrix[0, c] = runner.env.episode_total_delay_list
                    rainbow_proposed_erf_diff_context_matrix[1, c] = runner.env.total_energy_list
                    rainbow_proposed_erf_diff_context_matrix[2, c] = runner.env.acc_exp_list
                    rainbow_proposed_erf_diff_context_matrix[3, c] = runner.env.episode_acc_vio_num_list
                    rainbow_proposed_erf_diff_context_matrix[4, c] = runner.env.episode_re_trans_num_list
                    rainbow_proposed_erf_diff_context_matrix[5, c] = runner.env.reward_list
                    rainbow_proposed_erf_diff_context_matrix[6, c] = runner.env.episode_acc_vio_list
                    aver_min_acc = np.mean(runner.env.min_acc_list[0, :])

                elif algorithm == "rainbow_dqn" and env.name == "sse":
                    rainbow_sse_diff_context_matrix[0, c] = runner.env.episode_total_delay_list
                    rainbow_sse_diff_context_matrix[1, c] = runner.env.total_energy_list
                    rainbow_sse_diff_context_matrix[2, c] = runner.env.acc_exp_list
                    rainbow_sse_diff_context_matrix[3, c] = runner.env.episode_acc_vio_num_list
                    rainbow_sse_diff_context_matrix[4, c] = runner.env.episode_re_trans_num_list
                    rainbow_sse_diff_context_matrix[5, c] = runner.env.reward_list
                    rainbow_sse_diff_context_matrix[6, c] = runner.env.episode_acc_vio_list
                elif algorithm == "rainbow_dqn" and env.name == "tem":
                    rainbow_tem_diff_context_matrix[0, c] = runner.env.episode_total_delay_list
                    rainbow_tem_diff_context_matrix[1, c] = runner.env.total_energy_list
                    rainbow_tem_diff_context_matrix[2, c] = runner.env.acc_exp_list
                    rainbow_tem_diff_context_matrix[3, c] = runner.env.episode_acc_vio_num_list
                    rainbow_tem_diff_context_matrix[4, c] = runner.env.episode_re_trans_num_list
                    rainbow_tem_diff_context_matrix[5, c] = runner.env.reward_list
                    rainbow_tem_diff_context_matrix[6, c] = runner.env.episode_acc_vio_list
                elif algorithm == "dqn":
                    dqn_diff_context_matrix[0, c] = runner.env.episode_total_delay_list
                    dqn_diff_context_matrix[1, c] = runner.env.total_energy_list
                    dqn_diff_context_matrix[2, c] = runner.env.acc_exp_list
                    dqn_diff_context_matrix[3, c] = runner.env.episode_acc_vio_num_list
                    dqn_diff_context_matrix[4, c] = runner.env.episode_re_trans_num_list
                    dqn_diff_context_matrix[5, c] = runner.env.reward_list
                    dqn_diff_context_matrix[6, c] = runner.env.episode_acc_vio_list
                runner.env.reset()

    if scheme_name == "amac":
        # amac evaluation
        print("Evaluating AMAC")
        for c in range(len(context_list)):
            runner = amac.Amac(is_test=True)
            runner.context_list = [context_list[c]] * len(context_list)
            runner.run()
            action_str = "[1,2,3,4]"
            show_action_num = 3
            counter = Counter(runner.action_list)
            most_common_elements = counter.most_common(show_action_num)
            for act, freq in most_common_elements:
                action_freq = round(freq / runner.slot_num * 100, 2)
                if action_str == "[1,2,3,4]+":
                    action_str = action_str + act + "(" + str(action_freq) + "%)"
                else:
                    action_str = action_str + ";" + act + "(" + str(action_freq) + "%)"
            print(action_str)

            amac_diff_context_matrix[0, c] = runner.total_delay_list
            amac_diff_context_matrix[1, c] = runner.total_energy_list
            amac_diff_context_matrix[2, c] = runner.total_acc_list
            amac_diff_context_matrix[3, c] = runner.acc_vio_num
            amac_diff_context_matrix[4, c] = runner.re_trans_num_list
            amac_diff_context_matrix[5, c] = runner.step_reward_list
            amac_diff_context_matrix[6, c] = runner.episode_acc_vio_list

    if scheme_name == "proposed_erf":
        mat_name = "experiments/diff_context_data/rainbow_proposed_erf_diff_context_matrix.mat"
        savemat(mat_name, {"rainbow_proposed_erf_diff_context_matrix": rainbow_proposed_erf_diff_context_matrix,
                           "aver_min_acc": aver_min_acc,"context_list":context_list})
    elif scheme_name == "proposed_origin":
        mat_name = "experiments/diff_context_data/rainbow_proposed_origin_diff_context_matrix.mat"
        savemat(mat_name, {"rainbow_proposed_origin_diff_context_matrix": rainbow_proposed_origin_diff_context_matrix})
    elif scheme_name == "sse":
        mat_name = "experiments/diff_context_data/rainbow_sse_diff_context_matrix.mat"
        savemat(mat_name, {"rainbow_sse_diff_context_matrix": rainbow_sse_diff_context_matrix})
    elif scheme_name == "tem":
        mat_name = "experiments/diff_context_data/rainbow_tem_diff_context_matrix.mat"
        savemat(mat_name, {"rainbow_tem_diff_context_matrix": rainbow_tem_diff_context_matrix})
    elif scheme_name == "dqn":
        mat_name = "experiments/diff_context_data/dqn_diff_context_matrix.mat"
        savemat(mat_name, {"dqn_diff_context_matrix": dqn_diff_context_matrix})
    elif scheme_name == "amac":
        mat_name = "experiments/diff_context_data/amac_diff_context_matrix.mat"
        savemat(mat_name, {"amac_diff_context_matrix": amac_diff_context_matrix})

if __name__ == '__main__':
    time_start = time.time()
    run_single(40,"amac")
    # run_all(40)
    time_end = time.time()
    print("Running Time：" + str(time_end - time_start) + " Second")
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
time_start = time.time()

seed = 31
context_num = 3
show_action_num = 3
context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
algorithms = ["Proposed_erf", "Proposed_origin", "SSE", "TEM", "DQN", "AMAC"]
columns = ["Reward", "Delay", "Energy", "Accuracy", "Accuracy Vio Rate", "Re-trans Num", "Most Picked " + str(show_action_num) + " Action"]

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
random.seed(seed)
seed_torch(seed)

table_data = np.empty([len(context_list), len(algorithms), len(columns)],dtype='<U10000')


env_list = [ EnvProposed_erf(), EnvProposed_origin(), EnvSSE(), EnvTEM()]
env_num = len(set(type(obj) for obj in env_list))

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
        #     log_dir='runs/dqn/{}_env_{}_number_{}_seed_{}'.format(self.algorithm, env_name, number, seed))

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
                # terminal means dead or win,there is no next state s';
                # but when reaching the max_steps,there is a next state s' actually.
                if done and steps != self.args.limit:
                    terminal = True
                else:
                    terminal = False

                self.replay_buffer.store_transition(state, action, reward, next_state, terminal,
                                                    done)  # Store the transition
                state = next_state

                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)



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

if __name__ == '__main__':
    # Reward, Delay, Energy, Accuracy, Accuracy Vio, Re-trans, Actions
    # rainbow_proposed_erf_diff_context_matrix = np.zeros([7,len(context_list)],dtype=object)
    # rainbow_proposed_origin_diff_context_matrix = np.zeros([7,len(context_list)],dtype=object)
    # rainbow_sse_diff_context_matrix = np.zeros([7,len(context_list)],dtype=object)
    # rainbow_tem_diff_context_matrix = np.zeros([7,len(context_list)],dtype=object)
    # amac_diff_context_matrix = np.zeros([7,len(context_list)],dtype=object)
    # dqn_diff_context_matrix = np.zeros([7,len(context_list)],dtype=object)

    length = 3000  # Number of steps / episode
    number = 1  # Number of episode to train
    steps = number * length  # Total step number

    drl_algorithm_list = ["rainbow_dqn", "dqn"]
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

        for w in range(len(context_list)):
            scheme_id = 0
            for env_id in range(env_num):
                if algorithm == "dqn" and env_id != 1:
                    continue
                env_index = 0
                env = env_list[env_id]
                runner = Runner(args=args, env=env, number=1, seed=seed)
                folder_path = context_list[w] + "_models"

                runner.env.context_list = [context_list[w]] * len(context_list)
                # load the model
                runner.agent.net, runner.agent.target_net = sl.load_nn_model_diff_context(runner, folder_path)
                print("env name:", env.name)
                print("current context:", context_list[w])
                print("algorithm:", runner.algorithm)
                runner.run()

                # save the data
                delay_mean = np.mean(runner.env.total_delay_list)
                delay_std  = np.std(runner.env.total_delay_list)
                delay_str = str(delay_mean) + "\u00B1" + str(delay_std)
                table_data[w,scheme_id,0] = delay_str

                energy_mean = np.mean(runner.env.total_energy_list)
                energy_std  = np.std(runner.env.total_energy_list)
                energy_str = str(energy_mean) + "\u00B1" + str(energy_std)
                table_data[w,scheme_id,1] = energy_str

                acc_mean = np.mean(runner.env.acc_exp_list)
                acc_std  = np.std(runner.env.acc_exp_list)
                acc_str = str(acc_mean) + "\u00B1" + str(acc_std)
                table_data[w,scheme_id,2] = acc_str

                acc_vio_mean = np.mean(runner.env.episode_acc_vio_num_list)
                # acc_vio_std = np.std(runner.env.episode_acc_vio_num_list)
                acc_vio_str = str(acc_vio_mean)
                table_data[w, scheme_id, 3] = acc_vio_str

                re_trans_mean = np.mean(runner.env.episode_re_trans_num_list)
                # re_trans_std = np.std(runner.env.episode_re_trans_num_list)
                re_trans_str = str(re_trans_mean)
                table_data[w, scheme_id, 4] = re_trans_str

                reward_mean = np.mean(runner.env.reward_list)
                reward_std = np.std(runner.env.reward_list)
                reward_str = str(reward_mean) + "\u00B1" + str(reward_std)
                table_data[w, scheme_id, 5] = reward_str


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
                table_data[w, scheme_id, 6] = action_str
            scheme_id += 1
            runner.env.reset()

            # amac evaluation
            print("Evaluating AMAC")  # amac
            action_str = "[1,2,3,4]"
            runner = amac.Amac(is_test=True)
            runner.context_list = [context_list[w]] * len(context_list)
            runner.run()
            delay_mean = np.mean(runner.total_delay_list)
            delay_std = np.std(runner.total_delay_list)
            delay_str = str(delay_mean) + "\u00B1" + str(delay_std)
            table_data[w, scheme_id, 0] = delay_str

            energy_mean = np.mean(runner.total_energy_list)
            energy_std = np.std(runner.total_energy_list)
            energy_str = str(energy_mean) + "\u00B1" + str(energy_std)
            table_data[w, scheme_id, 1] = energy_str

            acc_mean = np.mean(runner.acc_exp_list)
            acc_std = np.std(runner.acc_exp_list)
            acc_str = str(acc_mean) + "\u00B1" + str(acc_std)
            table_data[w, scheme_id, 2] = acc_str

            acc_vio_mean = np.mean(runner.acc_vio_num_list)
            acc_vio_std = np.std(runner.acc_vio_num_list)
            acc_vio_str = str(acc_vio_mean) + "\u00B1" + str(acc_vio_std)
            table_data[w, scheme_id, 3] = acc_vio_str

            re_trans_mean = np.mean(runner.re_trans_num_list)
            re_trans_std = np.std(runner.re_trans_num_list)
            re_trans_str = str(re_trans_mean) + "\u00B1" + str(re_trans_std)
            table_data[w, scheme_id, 4] = re_trans_str

            reward_mean = np.mean(runner.reward_list)
            reward_std = np.std(runner.reward_list)
            reward_str = str(reward_mean) + "\u00B1" + str(reward_std)
            table_data[w, scheme_id, 5] = reward_str

            counter = Counter(runner.action_list)
            most_common_elements = counter.most_common(show_action_num)
            for act, freq in most_common_elements:
                action_freq = round(freq / runner.slot_num * 100, 2)
                if action_str == "[1,2,3,4]":
                    action_str = action_str + act + "(" + str(action_freq) + ")"
                else:
                    action_str = action_str + ";" + act + "(" + str(action_freq) + ")"
            print(action_str)
            table_data[w, scheme_id, 6] = action_str


    # output the table as an Excel file
    create_excel(table_data, "experiments/diff_context_data/performance_table_diff_context.xlsx")


    time_end = time.time()
    print("Running Time：" + str(time_end - time_start) + "Second")
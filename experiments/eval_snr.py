import torch
import numpy as np
import gym
from envs.env_proposed_origin import EnvProposed_origin
from envs.env_proposed_erf import EnvProposed_erf
from envs.env_tem import EnvTEM
from envs.env_sse import EnvSSE
from torch.utils.tensorboard import SummaryWriter
from rainbow_dqn.rainbow_replay_buffer import *
from rainbow_dqn.rainbow_agent import DQN
from baselines import amac
import argparse
import random
import tools.saving_loading as sl
import time
import os
from scipy.io import savemat

from experiments.eval_bad_actions import get_top_k_values
from collections import Counter


class Runner:
    def __init__(self, args, env, number, seed):
        self.args = args
        self.number = number
        self.seed = seed

        self.env = env
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = self.env.slot_num  # Maximum number of steps per episode
        print("env name:", self.env.name)
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

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
            episode_steps = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=self.epsilon)
                next_state, reward, done = self.env.step(action)
                self.total_steps += 1

                if not self.args.use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # terminal means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != self.args.episode_limit:
                    terminal = True
                else:
                    terminal = False

                self.replay_buffer.store_transition(state, action, reward, next_state, terminal,
                                                    done)  # Store the transition
                state = next_state

                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)




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
    # snr_db_list = np.arange(1.5, 4.25, 0.25)
    snr_db_list = np.arange(1, 3.5, 0.5)
    drl_algorithm_list = ["rainbow_dqn", "dqn"]
    env_list = [EnvProposed_erf(), EnvProposed_origin(), EnvSSE(), EnvTEM()]
    env_num = len(set(type(obj) for obj in env_list))

    rainbow_proposed_erf_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)
    rainbow_proposed_origin_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)
    rainbow_sse_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)
    rainbow_tem_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)
    amac_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)
    dqn_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)

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

        for s in range(len(snr_db_list)):
            scheme_id = 0
            for env_id in range(env_num):
                if algorithm == "dqn" and env.name != "proposed_erf":
                    continue
                env_index = 0
                env = env_list[env_id]
                print(env.context_train_list)
                runner = Runner(args=args, env=env, number=1, seed=seed)
                # load the model
                runner.agent.net, runner.agent.target_net = sl.load_nn_model(runner)
                runner.env.input_snr(snr_db_list[s])
                print("snr_db:", runner.env.target_snr_db)
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
                    rainbow_proposed_origin_diff_snr_matrix[0, s] = runner.env.episode_total_delay_list
                    rainbow_proposed_origin_diff_snr_matrix[1, s] = runner.env.total_energy_list
                    rainbow_proposed_origin_diff_snr_matrix[2, s] = runner.env.acc_exp_list
                    rainbow_proposed_origin_diff_snr_matrix[3, s] = runner.env.episode_acc_vio_num_list
                    rainbow_proposed_origin_diff_snr_matrix[4, s] = runner.env.episode_re_trans_num_list
                    rainbow_proposed_origin_diff_snr_matrix[5, s] = runner.env.reward_list
                    rainbow_proposed_origin_diff_snr_matrix[6, s] = runner.env.episode_acc_vio_list

                elif algorithm == "rainbow_dqn" and env.name == "proposed_erf":
                    rainbow_proposed_erf_diff_snr_matrix[0, s] = runner.env.episode_total_delay_list
                    rainbow_proposed_erf_diff_snr_matrix[1, s] = runner.env.total_energy_list
                    rainbow_proposed_erf_diff_snr_matrix[2, s] = runner.env.acc_exp_list
                    rainbow_proposed_erf_diff_snr_matrix[3, s] = runner.env.episode_acc_vio_num_list
                    rainbow_proposed_erf_diff_snr_matrix[4, s] = runner.env.episode_re_trans_num_list
                    rainbow_proposed_erf_diff_snr_matrix[5, s] = runner.env.reward_list
                    rainbow_proposed_erf_diff_snr_matrix[6, s] = runner.env.episode_acc_vio_list
                    aver_min_acc = np.mean(runner.env.min_acc_list[0,:])

                elif algorithm == "rainbow_dqn" and env.name == "sse":
                    rainbow_sse_diff_snr_matrix[0, s] = runner.env.episode_total_delay_list
                    rainbow_sse_diff_snr_matrix[1, s] = runner.env.total_energy_list
                    rainbow_sse_diff_snr_matrix[2, s] = runner.env.acc_exp_list
                    rainbow_sse_diff_snr_matrix[3, s] = runner.env.episode_acc_vio_num_list
                    rainbow_sse_diff_snr_matrix[4, s] = runner.env.episode_re_trans_num_list
                    rainbow_sse_diff_snr_matrix[5, s] = runner.env.reward_list
                    rainbow_sse_diff_snr_matrix[6, s] = runner.env.episode_acc_vio_list
                elif algorithm == "rainbow_dqn" and env.name == "tem":
                    rainbow_tem_diff_snr_matrix[0, s] = runner.env.episode_total_delay_list
                    rainbow_tem_diff_snr_matrix[1, s] = runner.env.total_energy_list
                    rainbow_tem_diff_snr_matrix[2, s] = runner.env.acc_exp_list
                    rainbow_tem_diff_snr_matrix[3, s] = runner.env.episode_acc_vio_num_list
                    rainbow_tem_diff_snr_matrix[4, s] = runner.env.episode_re_trans_num_list
                    rainbow_tem_diff_snr_matrix[5, s] = runner.env.reward_list
                    rainbow_tem_diff_snr_matrix[6, s] = runner.env.episode_acc_vio_list
                elif algorithm == "dqn":
                    dqn_diff_snr_matrix[0, s] = runner.env.episode_total_delay_list
                    dqn_diff_snr_matrix[1, s] = runner.env.total_energy_list
                    dqn_diff_snr_matrix[2, s] = runner.env.acc_exp_list
                    dqn_diff_snr_matrix[3, s] = runner.env.episode_acc_vio_num_list
                    dqn_diff_snr_matrix[4, s] = runner.env.episode_re_trans_num_list
                    dqn_diff_snr_matrix[5, s] = runner.env.reward_list
                    dqn_diff_snr_matrix[6, s] = runner.env.episode_acc_vio_list
                runner.env.reset()

    # amac evaluation
    print("Evaluating AMAC")
    for s in range(len(snr_db_list)):
        runner = amac.Amac(is_test=True)
        runner.target_snr_db = snr_db_list[s]
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

        amac_diff_snr_matrix[0, s] = runner.total_delay_list
        amac_diff_snr_matrix[1, s] = runner.total_energy_list
        amac_diff_snr_matrix[2, s] = runner.total_acc_list
        amac_diff_snr_matrix[3, s] = runner.acc_vio_num
        amac_diff_snr_matrix[4, s] = runner.re_trans_num_list
        amac_diff_snr_matrix[5, s] = runner.step_reward_list
        amac_diff_snr_matrix[6, s] = runner.episode_acc_vio_list

    # save all the data
    mat_name = "experiments/diff_snr_data/diff_snr_data.mat"
    savemat(mat_name,
            {"rainbow_proposed_erf_diff_snr_matrix": rainbow_proposed_erf_diff_snr_matrix,
             "rainbow_proposed_origin_diff_snr_matrix": rainbow_proposed_origin_diff_snr_matrix,
             "rainbow_sse_diff_snr_matrix": rainbow_sse_diff_snr_matrix,
             "rainbow_tem_diff_snr_matrix": rainbow_tem_diff_snr_matrix,
             "dqn_diff_snr_matrix": dqn_diff_snr_matrix,
             "amac_diff_snr_matrix": amac_diff_snr_matrix,
             "snr_db_list":snr_db_list,
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
    # snr_db_list = np.arange(1.5, 4.25, 0.25)
    snr_db_list = np.arange(1, 3.5, 0.5)

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

    rainbow_proposed_erf_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)
    rainbow_proposed_origin_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)
    rainbow_sse_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)
    rainbow_tem_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)
    amac_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)
    dqn_diff_snr_matrix = np.zeros([7, len(snr_db_list)], dtype=object)

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

        for s in range(len(snr_db_list)):
            scheme_id = 0
            for env_id in range(env_num):
                env_index = 0
                env = env_list[env_id]
                runner = Runner(args=args, env=env, number=1, seed=seed)
                # load the model
                runner.agent.net, runner.agent.target_net = sl.load_nn_model(runner)
                runner.env.input_snr(snr_db_list[s])
                print("snr_db:", runner.env.target_snr_db)
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
                    rainbow_proposed_origin_diff_snr_matrix[0, s] = runner.env.episode_total_delay_list
                    rainbow_proposed_origin_diff_snr_matrix[1, s] = runner.env.total_energy_list
                    rainbow_proposed_origin_diff_snr_matrix[2, s] = runner.env.acc_exp_list
                    rainbow_proposed_origin_diff_snr_matrix[3, s] = runner.env.episode_acc_vio_num_list
                    rainbow_proposed_origin_diff_snr_matrix[4, s] = runner.env.episode_re_trans_num_list
                    rainbow_proposed_origin_diff_snr_matrix[5, s] = runner.env.reward_list
                    rainbow_proposed_origin_diff_snr_matrix[6, s] = runner.env.episode_acc_vio_list

                elif algorithm == "rainbow_dqn" and env.name == "proposed_erf":
                    rainbow_proposed_erf_diff_snr_matrix[0, s] = runner.env.episode_total_delay_list
                    rainbow_proposed_erf_diff_snr_matrix[1, s] = runner.env.total_energy_list
                    rainbow_proposed_erf_diff_snr_matrix[2, s] = runner.env.acc_exp_list
                    rainbow_proposed_erf_diff_snr_matrix[3, s] = runner.env.episode_acc_vio_num_list
                    rainbow_proposed_erf_diff_snr_matrix[4, s] = runner.env.episode_re_trans_num_list
                    rainbow_proposed_erf_diff_snr_matrix[5, s] = runner.env.reward_list
                    rainbow_proposed_erf_diff_snr_matrix[6, s] = runner.env.episode_acc_vio_list
                    aver_min_acc = np.mean(runner.env.min_acc_list[0, :])

                elif algorithm == "rainbow_dqn" and env.name == "sse":
                    rainbow_sse_diff_snr_matrix[0, s] = runner.env.episode_total_delay_list
                    rainbow_sse_diff_snr_matrix[1, s] = runner.env.total_energy_list
                    rainbow_sse_diff_snr_matrix[2, s] = runner.env.acc_exp_list
                    rainbow_sse_diff_snr_matrix[3, s] = runner.env.episode_acc_vio_num_list
                    rainbow_sse_diff_snr_matrix[4, s] = runner.env.episode_re_trans_num_list
                    rainbow_sse_diff_snr_matrix[5, s] = runner.env.reward_list
                    rainbow_sse_diff_snr_matrix[6, s] = runner.env.episode_acc_vio_list
                elif algorithm == "rainbow_dqn" and env.name == "tem":
                    rainbow_tem_diff_snr_matrix[0, s] = runner.env.episode_total_delay_list
                    rainbow_tem_diff_snr_matrix[1, s] = runner.env.total_energy_list
                    rainbow_tem_diff_snr_matrix[2, s] = runner.env.acc_exp_list
                    rainbow_tem_diff_snr_matrix[3, s] = runner.env.episode_acc_vio_num_list
                    rainbow_tem_diff_snr_matrix[4, s] = runner.env.episode_re_trans_num_list
                    rainbow_tem_diff_snr_matrix[5, s] = runner.env.reward_list
                    rainbow_tem_diff_snr_matrix[6, s] = runner.env.episode_acc_vio_list
                elif algorithm == "dqn":
                    dqn_diff_snr_matrix[0, s] = runner.env.episode_total_delay_list
                    dqn_diff_snr_matrix[1, s] = runner.env.total_energy_list
                    dqn_diff_snr_matrix[2, s] = runner.env.acc_exp_list
                    dqn_diff_snr_matrix[3, s] = runner.env.episode_acc_vio_num_list
                    dqn_diff_snr_matrix[4, s] = runner.env.episode_re_trans_num_list
                    dqn_diff_snr_matrix[5, s] = runner.env.reward_list
                    dqn_diff_snr_matrix[6, s] = runner.env.episode_acc_vio_list
                runner.env.reset()

    if scheme_name == "amac":
        # amac evaluation
        print("Evaluating AMAC")
        for s in range(len(snr_db_list)):
            runner = amac.Amac(is_test=True)
            runner.target_snr_db = snr_db_list[s]
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

            amac_diff_snr_matrix[0, s] = runner.total_delay_list
            amac_diff_snr_matrix[1, s] = runner.total_energy_list
            amac_diff_snr_matrix[2, s] = runner.total_acc_list
            amac_diff_snr_matrix[3, s] = runner.acc_vio_num
            amac_diff_snr_matrix[4, s] = runner.re_trans_num_list
            amac_diff_snr_matrix[5, s] = runner.step_reward_list
            amac_diff_snr_matrix[6, s] = runner.episode_acc_vio_list

    if scheme_name == "proposed_erf":
        mat_name = "experiments/diff_snr_data/rainbow_proposed_erf_diff_snr_matrix.mat"
        savemat(mat_name, {"rainbow_proposed_erf_diff_snr_matrix": rainbow_proposed_erf_diff_snr_matrix,
                           "aver_min_acc": aver_min_acc})
    elif scheme_name == "proposed_origin":
        mat_name = "experiments/diff_snr_data/rainbow_proposed_origin_diff_snr_matrix.mat"
        savemat(mat_name, {"rainbow_proposed_origin_diff_snr_matrix": rainbow_proposed_origin_diff_snr_matrix})
    elif scheme_name == "sse":
        mat_name = "experiments/diff_snr_data/rainbow_sse_diff_snr_matrix.mat"
        savemat(mat_name, {"rainbow_sse_diff_snr_matrix": rainbow_sse_diff_snr_matrix})
    elif scheme_name == "tem":
        mat_name = "experiments/diff_snr_data/rainbow_tem_diff_snr_matrix.mat"
        savemat(mat_name, {"rainbow_tem_diff_snr_matrix": rainbow_tem_diff_snr_matrix})
    elif scheme_name == "dqn":
        mat_name = "experiments/diff_snr_data/dqn_diff_snr_matrix.mat"
        savemat(mat_name, {"dqn_diff_snr_matrix": dqn_diff_snr_matrix})
    elif scheme_name == "amac":
        mat_name = "experiments/diff_snr_data/amac_diff_snr_matrix.mat"
        savemat(mat_name, {"amac_diff_snr_matrix": amac_diff_snr_matrix})




if __name__ == '__main__':
    time_start = time.time()
    run_single(40,"dqn")
    time_end = time.time()
    print("Running Time：" + str(time_end - time_start) + " Second")



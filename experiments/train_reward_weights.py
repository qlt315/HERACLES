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
import argparse
import random
import tools.saving_loading as sl
import time
import os
time_start = time.time()
env_list = [EnvProposed_origin(), EnvProposed_erf(), EnvSSE(), EnvTEM()]
env_num = len(set(type(obj) for obj in env_list))

seed = 666
kappa_num = 3
kappa_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
random.seed(seed)
seed_torch(seed)


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

                # if self.total_steps % self.args.evaluate_freq == 0:
                #     self.evaluate_policy()

    # def evaluate_policy(self, ):
    #     evaluate_reward = 0
    #     self.agent.net.eval()
    #     for _ in range(self.args.evaluate_times):
    #         state = self.env_evaluate.reset()
    #         done = False
    #         episode_reward = 0
    #         while not done:
    #             action = self.agent.choose_action(state, epsilon=0)
    #             next_state, reward, done = self.env_evaluate.step(action)
    #             episode_reward += reward
    #             state = next_state
    #         evaluate_reward += episode_reward
    #     self.agent.net.train()
    #     evaluate_reward /= self.args.evaluate_times
    #     self.evaluate_rewards.append(evaluate_reward)
    #     print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward, self.epsilon))
    #     self.writer.add_scalar('step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)


if __name__ == '__main__':
    episode_length = 3000  # Number of steps / episode
    episode_number = 10  # Number of episode to train
    steps = episode_number * episode_length  # Total step number

    algorithm_list = ["dqn"]
    for algo_id in range(len(algorithm_list)):
        algorithm = algorithm_list[algo_id]
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
        for kappa_flag in range(kappa_num):
            for env_id in range(env_num):
                if algorithm == "dqn" and env_id != 1:
                    continue
                for w in range(len(kappa_list)):
                    env_index = 0
                    env = env_list[env_id]
                    runner = Runner(args=args, env=env, number=1, seed=seed)

                    if kappa_flag == 0:  # acc kappa
                        runner.env.kappa_1 = kappa_list[w]
                        runner.env.kappa_2 = 1
                        runner.env.kappa_3 = 1
                        folder_path = "kappa1_" + str(kappa_list[w]) + "_kappa2_1_kappa3_1"
                    elif kappa_flag == 1:  # delay kappa
                        runner.env.kappa_1 = 1
                        runner.env.kappa_2 = kappa_list[w]
                        runner.env.kappa_3 = 1
                        folder_path = "kappa1_1" + "_kappa2_" + str(kappa_list[w]) + "_kappa3_1"
                    elif kappa_flag == 2:
                        runner.env.kappa_1 = 1
                        runner.env.kappa_2 = 1
                        runner.env.kappa_3 = kappa_list[w]
                        folder_path = "kappa1_1_kappa2_1" + "_kappa3_" + str(kappa_list[w])

                    # Path for the new folder
                    if not os.path.exists("experiments/diff_reward_weights_data/" + folder_path):
                        os.makedirs("experiments/diff_reward_weights_data/" + folder_path)
                        print(f'Folder "{folder_path}" has been created.')
                    else:
                        print(f'Folder "{folder_path}" already exists.')

                    print("algorithm:", runner.algorithm)
                    runner.run()

                    # save the model
                    sl.save_nn_model_diff_kappa(runner,folder_path)
                    runner.env.reset()

    time_end = time.time()
    print("Running Time：" + str(time_end - time_start) + "Second")
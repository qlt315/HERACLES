import random
from scipy.io import savemat
import numpy as np
import torch
from train_envs.env_proposed import EnvProposed
from baselines.random_agent import RandomAgent


is_test = False # False for plotting the reward, True for eval data

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if is_test:
    seed_list = [37]
else:
    seed_list = [666, 555, 444, 333, 111]
# seed_list = [666]
# parameters
episode_length = 3000  # Number of steps / episode
episode_number = 20  # Number of episode to train
num_frames = episode_number * episode_length  # Total step number
memory_size = 10000
batch_size = 128
target_update = 100

step_reward_matrix = np.zeros([len(seed_list), int(num_frames)])

for k in range(len(seed_list)):
    seed = seed_list[k]
    print("Train index:", k, "seed", seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    # environment
    # env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")
    env = EnvProposed()
    env.name = "random"


    # train
    agent = RandomAgent(env, memory_size, batch_size, target_update, seed)

    agent.train(num_frames)

    step_reward_matrix[k, :] = np.array(env.step_reward_list)

    env.step_reward_list = []
    env.reset()

if is_test:
    mat_name ="baselines/eval_random_data.mat"
    print("Env Name:", mat_name)
    savemat(mat_name,
            {env.name + "_eval_episode_total_delay": np.sum(env.episode_total_delay_list) / episode_number,
             env.name + "_eval_episode_total_energy": np.sum(env.episode_total_energy_list) / episode_number,
             env.name + "_eval_episode_reward": np.sum(env.episode_reward_list) / episode_number,
             env.name + "_eval_episode_acc_exp": np.sum(env.episode_acc_exp_list) / episode_number,
             env.name + "_eval_episode_delay_vio_num": np.sum(env.episode_delay_vio_num_list) / episode_number,
             env.name + "_eval_episode_remain_energy": np.sum(env.episode_remain_energy_list) / episode_number,
             env.name + "_eval_episode_re_trans_number": np.sum(env.episode_re_trans_num_list) / episode_number,
             })
else:
    # save the reward data
    mat_name = "baselines/random_reward.mat"
    print("Env Name:", mat_name)
    savemat(mat_name, {"random_step_reward_matrix": step_reward_matrix})


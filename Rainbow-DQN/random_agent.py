import random
from scipy.io import savemat
from envs.env_random import EnvRandom
import numpy as np
import torch
from agent import DQNAgent

# environment
# env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")
env = EnvRandom()
env.name = "random"


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


seed_list = [666, 555, 444, 333, 111]
# seed_list = [666]
# parameters
episode_length = env.slot_num  # Number of steps / episode
episode_number = 1  # Number of episode to train
num_frames = episode_number * episode_length  # Total step number
memory_size = 10000
batch_size = 128000000000
target_update = 100000000  # so the agent will randomly select the action without training

step_reward_matrix = np.zeros([len(seed_list), int(num_frames)])

for k in range(len(seed_list)):
    seed = seed_list[k]
    print("Train index:", k, "seed", seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    # train
    agent = DQNAgent(env, memory_size, batch_size, target_update, seed)

    agent.train(num_frames)

    step_reward_matrix[k, :] = np.array(env.step_reward_list)

    env.step_reward_list = []
    env.reset()

# save the  data
mat_name = "eval_data/eval_random_data.mat"
print("Env Name:", mat_name)
savemat(mat_name,
        {env.name + "_train_episode_total_delay": env.episode_total_delay_list,
         env.name + "_eval_episode_total_energy": env.episode_total_energy_list,
         env.name + "_eval_episode_reward": env.episode_reward_list,
         env.name + "_eval_episode_acc_exp": env.episode_acc_exp_list,
         env.name + "_eval_episode_delay_vio_num": env.episode_delay_vio_num_list,
         env.name + "_eval_episode_remain_energy": env.episode_remain_energy_list,
         env.name + "_eval_episode_re_trans_number": env.episode_re_trans_num_list,
         "random_step_reward_matrix": step_reward_matrix
         })

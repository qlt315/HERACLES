import random
from scipy.io import savemat
from envs.env_proposed import EnvProposed
import numpy as np
import torch
import time
from rainbow_agent import DQNAgent

time_start = time.time()



def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


seed_list = [666, 555, 444, 333, 111]
# seed_list = [666]
episode_length = 3000  # Number of steps / episode
episode_number = 20  # Number of episode to train
num_frames = episode_number * episode_length  # Total step number
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
    # env = EnvSSE()
    # env = EnvTEM()

    # parameters

    memory_size = 10000
    batch_size = 128
    target_update = 100


    # train
    agent = DQNAgent(env, memory_size, batch_size, target_update, seed)

    agent.train(num_frames)

    # save the data
    if env.name == "proposed":
        mat_name = "rainbow_dqn/train_data/train_proposed_data.mat"
    elif env.name == "sse":
        mat_name = "rainbow_dqn/train_data/train_sse_data.mat"
    elif env.name == "tem":
        mat_name = "rainbow_dqn/train_data/train_tem_data.mat"
    print("Env Name:", mat_name)
    savemat(mat_name,
            {"rainbow_dqn"+env.name+"_train_episode_total_delay": env.episode_total_delay_list,
             "rainbow_dqn"+env.name+"_train_episode_total_energy": env.episode_total_energy_list,
             "rainbow_dqn"+env.name+"_train_episode_reward": env.episode_reward_list,
             "rainbow_dqn"+env.name+"_train_episode_acc_exp": env.episode_acc_exp_list,
             "rainbow_dqn"+env.name+"_train_episode_remain_energy": env.episode_remain_energy_list,
             "rainbow_dqn"+env.name+"_train_episode_re_trans_number": env.episode_re_trans_num_list,
             "rainbow_dqn"+env.name+"_train_step_reward": env.step_reward_list
             })
    step_reward_matrix[k, :] = np.array(env.step_reward_list)

    # save the model
    if env.name == "proposed":
        torch.save(agent.dqn, 'rainbow_dqn/models/dqn_proposed.pth')
        torch.save(agent.dqn_target, 'rainbow_dqn/models/rainbow_dqn_target_proposed.pth')
    elif env.name == "sse":
        torch.save(agent.dqn, 'rainbow_dqn/models/dqn_sse.pth')
        torch.save(agent.dqn_target, 'rainbow_dqn/models/rainbow_dqn_target_sse.pth')
    elif env.name == "tem":
        torch.save(agent.dqn, 'rainbow_dqn/models/dqn_tem.pth')
        torch.save(agent.dqn_target, 'rainbow_dqn/models/rainbow_dqn_target_tem.pth')

    env.step_reward_list = []
    env.reset()

# save the step reward data
if env.name == "proposed":
    reward_mat_name = "rainbow_dqn/train_data/train_proposed_step_reward_matrix.mat"
    matrix_name = "rainbow_dqn_train_proposed_step_reward_matrix"
elif env.name == "sse":
    reward_mat_name = "rainbow_dqn/train_data/train_sse_step_reward_matrix.mat"
    matrix_name = "rainbow_dqn_train_sse_step_reward_matrix"
elif env.name == "tem":
    reward_mat_name = "rainbow_dqn/train_data/train_tem_step_reward_matrix.mat"
    matrix_name = "rainbow_dqn_train_tem_step_reward_matrix"

savemat(reward_mat_name,
        {matrix_name: step_reward_matrix,
         })

time_end = time.time()
print("Running Time："+str(time_end - time_start)+"Second")
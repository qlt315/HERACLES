import random
from scipy.io import savemat
from envs.env_proposed import EnvProposed
import numpy as np
import torch
from rainbow_agent import DQNAgent

test_seed = 37


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(test_seed)
random.seed(test_seed)
seed_torch(test_seed)


eval_time = 1
memory_size = 10000
batch_size = 128
target_update = 100



eval_total_delay_list = np.zeros([1, eval_time])
eval_total_energy_list = np.zeros([1, eval_time])
eval_reward_list = np.zeros([1, eval_time])
eval_acc_exp_list = np.zeros([1, eval_time])
eval_delay_vio_num_list = np.zeros([1, eval_time])
eval_remain_energy_list = np.zeros([1, eval_time])
eval_re_trans_number_list = np.zeros([1, eval_time])

for i in range(eval_time):
    # env = EnvSSE()
    # env = EnvTEM()
    env = EnvProposed()
    # load the model
    test_agent = DQNAgent(env, memory_size, batch_size, target_update, test_seed)
    if env.name == "proposed":
        test_agent.dqn = torch.load('rainbow_dqn/models/rainbow_dqn_proposed.pth')
        test_agent.dqn_target = torch.load('rainbow_dqn/models/rainbow_dqn_target_proposed.pth')
    elif env.name == "sse":
        test_agent.dqn = torch.load('rainbow_dqn/models/rainbow_dqn_sse.pth')
        test_agent.dqn_target = torch.load('rainbow_dqn/models/rainbow_dqn_target_sse.pth')
    elif env.name == "tem":
        test_agent.dqn = torch.load('rainbow_dqn/models/rainbow_dqn_tem.pth')
        test_agent.dqn_target = torch.load('rainbow_dqn/models/rainbow_dqn_target_tem.pth')

    # save the data
    if env.name == "proposed":
        mat_name = "rainbow_dqn/eval_data/rainbow_dqn_eval_proposed_data.mat"
    elif env.name == "sse":
        mat_name = "rainbow_dqn/eval_data/rainbow_dqn_eval_sse_data.mat"
    elif env.name == "tem":
        mat_name = "rainbow_dqn/eval_data/rainbow_dqn_eval_tem_data.mat"

    test_agent.test(env, test_seed)
    eval_total_delay_list[0, i] = env.episode_total_delay_list[0]
    eval_total_energy_list[0, i] = env.episode_total_energy_list[0]
    eval_reward_list[0, i] = env.episode_reward_list[0]
    eval_acc_exp_list[0, i] = env.episode_acc_exp_list[0]
    eval_delay_vio_num_list[0, i] = env.episode_delay_vio_num_list[0]
    eval_remain_energy_list[0, i] = env.episode_remain_energy_list[0]
    eval_re_trans_number_list[0, i] = env.episode_re_trans_num_list[0]

savemat(mat_name,
        {"rainbow_dqn"+env.name+"_eval_episode_total_delay": np.mean(eval_total_delay_list),
         "rainbow_dqn"+env.name+"_eval_episode_total_energy": np.mean(eval_total_energy_list),
         "rainbow_dqn"+env.name+"_eval_episode_reward": np.mean(eval_reward_list),
         "rainbow_dqn"+env.name+"_eval_episode_acc_exp": np.mean(eval_acc_exp_list),
         "rainbow_dqn"+env.name+"_eval_episode_delay_vio_num": np.mean(eval_delay_vio_num_list),
         "rainbow_dqn"+env.name+"_eval_episode_remain_energy": np.mean(eval_remain_energy_list),
         "rainbow_dqn"+env.name+"_eval_episode_re_trans_number": np.mean(eval_re_trans_number_list)
         })

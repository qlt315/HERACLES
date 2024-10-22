import torch 
from scipy.io import savemat

def load_nn_model(runner):
    if runner.algorithm == "rainbow_dqn":
        if runner.env.name == "proposed_origin":
            net = torch.load('rainbow_dqn/models/rainbow_dqn_proposed_origin.pth')
            target_net = torch.load('rainbow_dqn/models/rainbow_dqn_target_proposed_origin.pth')
        elif runner.env.name == "proposed_erf":
            net = torch.load('rainbow_dqn/models/rainbow_dqn_proposed_erf.pth')
            target_net = torch.load('rainbow_dqn/models/rainbow_dqn_target_proposed_erf.pth')
        elif runner.env.name == "sse":
            net = torch.load('rainbow_dqn/models/rainbow_dqn_sse.pth')
            target_net = torch.load('rainbow_dqn/models/rainbow_dqn_target_sse.pth')
        elif runner.env.name == "tem":
            net = torch.load('rainbow_dqn/models/rainbow_dqn_tem.pth')
            target_net = torch.load('rainbow_dqn/models/rainbow_dqn_target_tem.pth')
    elif runner.algorithm == "dqn":
        if runner.env.name == "proposed_origin":
            net = torch.load('baselines/dqn/models/rainbow_dqn_proposed_origin.pth')
            target_net = torch.load('baselines/dqn/models/rainbow_dqn_target_proposed_origin.pth')
        elif runner.env.name == "proposed_erf":
            net = torch.load('baselines/dqn/models/rainbow_dqn_proposed_erf.pth')
            target_net = torch.load('baselines/dqn/models/rainbow_dqn_target_proposed_erf.pth')
        elif runner.env.name == "sse":
            net = torch.load('baselines/dqn/models/rainbow_dqn_sse.pth')
            target_net = torch.load('baselines/dqn/models/rainbow_dqn_target_sse.pth')
        elif runner.env.name == "tem":
            net = torch.load('baselines/dqn/models/rainbow_dqn_tem.pth')
            target_net = torch.load('baselines/dqn/models/rainbow_dqn_target_tem.pth')
    return net, target_net

def save_nn_model(runner):
    if runner.algorithm == "rainbow_dqn":
        if runner.env.name == "proposed_origin":
            torch.save(runner.agent.net, 'rainbow_dqn/models/rainbow_dqn_proposed_origin.pth')
            torch.save(runner.agent.target_net, 'rainbow_dqn/models/rainbow_dqn_target_proposed_origin.pth')
        elif runner.env.name == "proposed_erf":
            torch.save(runner.agent.net, 'rainbow_dqn/models/rainbow_dqn_proposed_erf.pth')
            torch.save(runner.agent.target_net, 'rainbow_dqn/models/rainbow_dqn_target_proposed_erf.pth')
        elif runner.env.name == "sse":
            torch.save(runner.agent.net, 'rainbow_dqn/models/rainbow_dqn_sse.pth')
            torch.save(runner.agent.target_net, 'rainbow_dqn/models/rainbow_dqn_target_sse.pth')
        elif runner.env.name == "tem":
            torch.save(runner.agent.net, 'rainbow_dqn/models/rainbow_dqn_tem.pth')
            torch.save(runner.agent.target_net, 'rainbow_dqn/models/rainbow_dqn_target_tem.pth')
    elif runner.algorithm == "dqn":
        if runner.env.name == "proposed_origin":
            torch.save(runner.agent.net, 'baselines/dqn/models/rainbow_dqn_proposed_origin".pth')
            torch.save(runner.agent.target_net, 'baselines/dqn/models/rainbow_dqn_target_proposed_origin.pth')
        elif runner.env.name == "proposed_erf":
            torch.save(runner.agent.net, 'baselines/dqn/models/rainbow_dqn_proposed_erf.pth')
            torch.save(runner.agent.target_net, 'baselines/dqn/models/rainbow_dqn_target_proposed_erf.pth')
        elif runner.env.name == "sse":
            torch.save(runner.agent.net, 'baselines/dqn/models/rainbow_dqn_sse.pth')
            torch.save(runner.agent.target_net, 'baselines/dqn/models/rainbow_dqn_target_sse.pth')
        elif runner.env.name == "tem":
            torch.save(runner.agent.net, 'baselines/dqn/models/rainbow_dqn_tem.pth')
            torch.save(runner.agent.target_net, 'baselines/dqn/models/rainbow_dqn_target_tem.pth')


def save_nn_model_diff_kappa(runner,folder_path):
    if runner.algorithm == "rainbow_dqn":
        if runner.env.name == "proposed_origin":
            torch.save(runner.agent.net, 'experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_proposed_origin.pth')
            torch.save(runner.agent.target_net, 'experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_target_proposed_origin.pth')
        elif runner.env.name == "proposed_erf":
            torch.save(runner.agent.net, 'experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_proposed_erf.pth')
            torch.save(runner.agent.target_net, 'experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_proposed_target_erf.pth')
        elif runner.env.name == "sse":
            torch.save(runner.agent.net, 'experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_sse.pth')
            torch.save(runner.agent.target_net, 'experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_target_sse.pth')
        elif runner.env.name == "tem":
            torch.save(runner.agent.net,  'experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_tem.pth')
            torch.save(runner.agent.target_net, 'experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_target_tem.pth')
    elif runner.algorithm == "dqn":
        if runner.env.name == "proposed_origin":
            torch.save(runner.agent.net, 'experiments/diff_reward_weights_data/' + folder_path + '/dqn_proposed_origin.pth')
            torch.save(runner.agent.target_net, 'experiments/diff_reward_weights_data/' + folder_path + '/dqn_target_proposed_origin.pth')
        elif runner.env.name == "proposed_erf":
            torch.save(runner.agent.net, 'experiments/diff_reward_weights_data/' + folder_path + '/dqn_proposed_erf.pth')
            torch.save(runner.agent.target_net, 'experiments/diff_reward_weights_data/' + folder_path + '/dqn_proposed_target_erf.pth')
        elif runner.env.name == "sse":
            torch.save(runner.agent.net, 'experiments/diff_reward_weights_data/' + folder_path + '/dqn_sse.pth')
            torch.save(runner.agent.target_net, 'experiments/diff_reward_weights_data/' + folder_path + '/dqn_target_sse.pth')
        elif runner.env.name == "tem":
            torch.save(runner.agent.net,  'experiments/diff_reward_weights_data/' + folder_path + '/dqn_tem.pth')
            torch.save(runner.agent.target_net, 'experiments/diff_reward_weights_data/' + folder_path + '/dqn_target_tem.pth')


def load_nn_model_diff_kappa(runner,folder_path):
    if runner.algorithm == "rainbow_dqn":
        if runner.env.name == "proposed_origin":
            net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_proposed_origin.pth')
            target_net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_target_proposed_origin.pth')
        elif runner.env.name == "proposed_erf":
            net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_proposed_erf.pth')
            target_net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_proposed_target_erf.pth')
        elif runner.env.name == "sse":
            net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_sse.pth')
            target_net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_target_sse.pth')
        elif runner.env.name == "tem":
            net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_tem.pth')
            target_net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/rainbow_dqn_target_tem.pth')
    elif runner.algorithm == "dqn":
        if runner.env.name == "proposed_origin":
            net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/dqn_proposed_erf.pth')
            target_net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/dqn_target_proposed_origin.pth')
        elif runner.env.name == "proposed_erf":
            net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/dqn_proposed_erf.pth')
            target_net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/dqn_proposed_target_erf.pth')
        elif runner.env.name == "sse":
            net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/dqn_sse.pth')
            target_net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/dqn_target_sse.pth')
        elif runner.env.name == "tem":
            net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/dqn_tem.pth')
            target_net = torch.load('experiments/diff_reward_weights_data/' + folder_path + '/dqn_target_tem.pth')
    return net, target_net

def save_train_data(runner,step_reward_matrix):
    # save the data
    if runner.algorithm == "rainbow_dqn":
        if runner.env.name == "proposed_origin":
            mat_name = "rainbow_dqn/train_data/train_proposed_origin_data.mat"
        elif runner.env.name == "proposed_erf":
            mat_name = "rainbow_dqn/train_data/train_proposed_erf_data.mat"
        elif runner.env.name == "sse":
            mat_name = "rainbow_dqn/train_data/train_sse_data.mat"
        elif runner.env.name == "tem":
            mat_name = "rainbow_dqn/train_data/train_tem_data.mat"

    elif runner.algorithm == "dqn":
        if runner.env.name == "proposed_origin":
            mat_name = "baselines/dqn/train_data/train_proposed_origin_data.mat"
        elif runner.env.name == "proposed_erf":
            mat_name = "baselines/dqn/train_data/train_proposed_erf_data.mat"
        elif runner.env.name == "sse":
            mat_name = "baselines/dqn/train_data/train_sse_data.mat"
        elif runner.env.name == "tem":
            mat_name = "baselines/dqn/train_data/train_tem_data.mat"

    savemat(mat_name,
            {runner.algorithm + "_" + runner.env.name + "_train_episode_total_delay": runner.env.episode_total_delay_list,
             runner.algorithm + "_" + runner.env.name + "_train_episode_total_energy": runner.env.episode_total_energy_list,
             runner.algorithm + "_" + runner.env.name + "_train_episode_reward": runner.env.episode_reward_list,
             runner.algorithm + "_" + runner.env.name + "_train_episode_acc_exp": runner.env.episode_acc_exp_list,
             runner.algorithm + "_" + runner.env.name + "_train_episode_remain_energy": runner.env.episode_remain_energy_list,
             runner.algorithm + "_" + runner.env.name + "_train_episode_re_trans_number": runner.env.episode_re_trans_num_list,
             runner.algorithm + "_" + runner.env.name + "_train_delay_vio_number": runner.env.episode_delay_vio_num_list,
             runner.algorithm + "_" + runner.env.name + "_train_acc_vio_number": runner.env.episode_acc_vio_num_list,
             runner.algorithm + "_" + runner.env.name + "_step_reward_matrix": step_reward_matrix
             })


def save_eval_data(runner):
    # save the data
    if runner.algorithm == "rainbow_dqn":
        if runner.env.name == "proposed_origin":
            mat_name = "rainbow_dqn/eval_data/eval_proposed_origin_data.mat"
        elif runner.env.name == "proposed_erf":
            mat_name = "rainbow_dqn/eval_data/eval_proposed_erf_data.mat"
        elif runner.env.name == "sse":
            mat_name = "rainbow_dqn/eval_data/eval_sse_data.mat"
        elif runner.env.name == "tem":
            mat_name = "rainbow_dqn/eval_data/eval_tem_data.mat"

    elif runner.algorithm == "dqn":
        if runner.env.name == "proposed_origin":
            mat_name = "baselines/dqn/eval_data/eval_proposed_origin_data.mat"
        elif runner.env.name == "proposed_erf":
            mat_name = "baselines/dqn/eval_data/eval_proposed_erf_data.mat"
        elif runner.env.name == "sse":
            mat_name = "baselines/dqn/eval_data/eval_sse_data.mat"
        elif runner.env.name == "tem":
            mat_name = "baselines/dqn/eval_data/eval_tem_data.mat"

    savemat(mat_name,
            {runner.algorithm + "_" + runner.env.name + "_eval_episode_total_delay": runner.env.episode_total_delay_list,
             runner.algorithm + "_" + runner.env.name + "_eval_episode_total_energy": runner.env.episode_total_energy_list,
             runner.algorithm + "_" + runner.env.name + "_eval_episode_reward": runner.env.episode_reward_list,
             runner.algorithm + "_" + runner.env.name + "_eval_episode_acc_exp": runner.env.episode_acc_exp_list,
             runner.algorithm + "_" + runner.env.name + "_eval_episode_remain_energy": runner.env.episode_remain_energy_list,
             runner.algorithm + "_" + runner.env.name + "_eval_episode_re_trans_number": runner.env.episode_re_trans_num_list,
             runner.algorithm + "_" + runner.env.name + "_eval_delay_vio_number": runner.env.episode_delay_vio_num_list,
             runner.algorithm + "_" + runner.env.name + "_eval_acc_vio_number": runner.env.episode_acc_vio_num_list,
             })
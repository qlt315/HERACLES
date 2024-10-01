import numpy as np
import os
import re
import scipy.io as sio
import random
import math
import envs.utils_proposed as util
from scipy.io import savemat

seed_list = [666, 555, 444, 333, 111]

eval_total_delay_list = np.zeros([1, len(seed_list)])
eval_total_energy_list = np.zeros([1, len(seed_list)])
eval_reward_list = np.zeros([1, len(seed_list)])
eval_acc_exp_list = np.zeros([1, len(seed_list)])
eval_delay_vio_num_list = np.zeros([1, len(seed_list)])
eval_remain_energy_list = np.zeros([1, len(seed_list)])
eval_re_trans_number_list = np.zeros([1, len(seed_list)])

slot_num = 9000  # Number of time slots
reward_list = np.zeros([len(seed_list), slot_num])
sensor_num = 4  # Number of sensors
bandwidth = 20e6  # System bandwidth (Hz)
max_power = 1  # Maximum transmit power (W)
Est_err_para = 0.5

kappa_1 = 1
kappa_2 = 1
enable_re_trans = True
sub_block_length = 128
target_snr_db = 10
for k in range(len(seed_list)):
    print("seed:",seed_list[k])
    max_energy = 6000  # Maximum energy consumption (J)
    curr_energy = max_energy  # Available energy of current slot
    last_energy = 0  # Consumed energy of last slot

    seed = seed_list[k]
    # Parameter settings
    np.random.seed(seed)

    # Channel generation
    h = (np.random.randn(slot_num) + 1j * np.random.randn(slot_num)) / np.sqrt(2)
    Est_err = Est_err_para * np.abs(h)
    hEst = h + Est_err + Est_err * 1j

    # Quantized data size
    data_size = np.zeros([1, sensor_num])
    for i in range(sensor_num):
        data_size[0, i] = np.random.rand() * 5000
    # Initialize lists
    total_delay_list = np.zeros([1, slot_num])
    total_energy_list = np.zeros([1, slot_num])
    total_acc_list = np.zeros([1, slot_num])

    delay_vio_num = 0
    re_trans_num = np.nan
    context_flag = 0
    consider_re_trans = 0  # 0 or 1

    context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
    context_prob = [0.05, 0.05, 0.2, 0.1, 0.2, 0.4]
    curr_context = None
    context_interval = 100  # Interval for context to change
    context_num = int(slot_num / context_interval)
    context_train_list = np.random.choice(list(range(len(context_list))),
                                          size=context_num, p=context_prob)
    # Data loading and fitting
    platform_data = sio.loadmat('system_data/platform_data.mat')

    folder_path = "system_data/typical modulation"
    file_list = os.listdir(folder_path)
    all_valid_file = []

    # Search for valid files
    for file_name in file_list:
        if f"esterr_{Est_err_para}" in file_name:
            all_valid_file.append(file_name)

    all_p = []  # Polynomial coefficients
    all_mod_method = []
    all_rate = []

    # Process valid files
    for file_name in all_valid_file:
        pattern = r'snr_([\d\._]+)_(\w+)_esterr_([\d\.]+)_rate_(\d+)_(\d+)'
        match = re.search(pattern, file_name)

        if match:
            snr_str = match.group(1)
            snr_values = list(map(float, snr_str.split('_')))
            snr_min, snr_int, snr_max = snr_values[0], snr_values[1], snr_values[2]
            mod_method = match.group(2)
            rate_num = int(match.group(4))
            rate_den = int(match.group(5))
            rate = rate_num / rate_den

            # Load BER data and fit
            ber_data = sio.loadmat(os.path.join(folder_path, file_name))
            snr_list = np.arange(snr_min, snr_max + snr_int, snr_int)
            p = np.polyfit(snr_list, ber_data['bler'].flatten(), 5)

            all_p.append(p)
            all_mod_method.append(mod_method)
            all_rate.append(rate)

    # Main simulation loop
    for i in range(slot_num):
        print("slot num:", i)
        noise_power = abs(hEst[i]) * max_power / (10 ** (target_snr_db / 10))
        curr_context_id = context_train_list[context_flag]
        curr_context = context_list[curr_context_id]
        if i % context_interval == 0 and i != 0:
            context_flag = context_flag + 1
            curr_context_id = context_train_list[context_flag]
            curr_context = context_list[curr_context_id]
        curr_energy = curr_energy - last_energy
        max_delay = 0.5  # Maximum tolerant delay (s)

        # Calculate SNR
        snr = np.abs(hEst[i]) * max_power / noise_power

        # Search for optimal transmission scheme

        reward_curr_list = []
        delay_curr_list = []
        energy_curr_list = []

        for j in range(len(all_valid_file)):
            p_curr = all_p[j]
            mod_method_curr = all_mod_method[j]
            rate_curr = all_rate[j]

            # BER Calculation
            snr_dB = 10 * np.log10(snr)
            ber_curr = np.polyval(p_curr, snr_dB)
            ber_curr = np.clip(ber_curr, 0, 1)
            acc_curr = platform_data[curr_context][22, 0]

            # Delay Calculation
            trans_rate = bandwidth * np.log2(1 + snr)
            # print("trans rate:",trans_rate)
            coded_data_size = math.floor(np.sum(data_size) / rate_curr)

            trans_delay = coded_data_size / trans_rate
            trans_energy = trans_delay * max_power

            # Placeholder for stem and branch computation
            stem_com_delay = np.sum(platform_data["stem_delay"])
            stem_com_energy = np.sum(platform_data["stem_energy"])
            branch_com_delay = platform_data["branch_delay"][22, 0]
            branch_com_energy = platform_data["branch_energy"][22, 0]

            # Retransmission (if enabled)
            if enable_re_trans:
                re_trans_num = 0
                block_num = np.floor(coded_data_size / sub_block_length)
                per_curr = 1 - (1 - ber_curr) ** sub_block_length
                # print("Using TM")
                # print("BER:",ber_curr)
                # print("PER:", per_curr)
                for l in range(int(block_num)):
                    re_trans_num_block = 0
                    is_trans_success = 0
                    while is_trans_success == 0:
                        # Generate 1 (success) with probability 1-p and 0 (fail) with p
                        is_trans_success = \
                            random.choices([0, 1], weights=[per_curr, 1 - per_curr])[0]
                        if is_trans_success == 1:
                            continue
                        else:
                            re_trans_num_block = re_trans_num_block + 1
                            if re_trans_num_block >= 100:
                                break
                    re_trans_num = re_trans_num + re_trans_num_block
                re_trans_delay = re_trans_num * (sub_block_length / trans_rate)
                trans_delay = trans_delay + re_trans_delay

            total_delay_list[0, i] = stem_com_energy.item() + trans_delay.item() + branch_com_delay.item()
            total_energy_list[0, i] = stem_com_energy.item() + trans_energy.item() + branch_com_energy.item()
            last_energy = total_energy_list[0, i]

            total_acc_list[0, i] = acc_curr

            if total_delay_list[0, i] > max_delay:
                delay_vio_num += 1

            acc_curr = util.acc_normalize(acc_curr, curr_context)
            reward_1 = acc_curr
            reward_2 = (max_delay - total_delay_list[0, i]) / max_delay
            reward_3 = total_energy_list[0, i]

            reward = reward_1 + kappa_1 * reward_2 - kappa_2 * reward_3
            reward_curr_list.append(reward)

        # Optimal scheme selection (select the one with the largest reward)

        opt_index = reward_curr_list.index(max(reward_curr_list))
        delay_vio_num = 0
        p_curr = all_p[opt_index]
        mod_method_curr = all_mod_method[opt_index]
        rate_curr = all_rate[opt_index]

        # BER Calculation
        snr_dB = 10 * np.log10(snr)
        ber_curr = np.polyval(p_curr, snr_dB)
        ber_curr = np.clip(ber_curr, 0, 1)
        acc_curr = platform_data[curr_context][22, 0]

        # Delay Calculation
        trans_rate = bandwidth * np.log2(1 + snr)
        # print("trans rate:",trans_rate)
        coded_data_size = math.floor(np.sum(data_size) / rate_curr)

        trans_delay = coded_data_size / trans_rate
        trans_energy = trans_delay * max_power

        stem_com_delay = np.sum(platform_data["stem_delay"])
        stem_com_energy = np.sum(platform_data["stem_energy"])
        branch_com_delay = platform_data["branch_delay"][22, 0]
        branch_com_energy = platform_data["branch_energy"][22, 0]

        # Retransmission (if enabled)
        if enable_re_trans:
            re_trans_num = 0
            block_num = np.floor(coded_data_size / sub_block_length)
            per_curr = 1 - (1 - ber_curr) ** sub_block_length
            # print("Using TM")
            # print("BER:",ber_curr)
            # print("PER:", per_curr)
            for l in range(int(block_num)):
                is_trans_success = 0
                while is_trans_success == 0:
                    # Generate 1 (success) with probability 1-p and 0 (fail) with p
                    is_trans_success = \
                        random.choices([0, 1], weights=[per_curr, 1 - per_curr])[0]
                    if is_trans_success == 1:
                        continue
                    else:
                        re_trans_num = re_trans_num + 1
            re_trans_delay = re_trans_num * (sub_block_length / trans_rate)
            trans_delay = trans_delay + re_trans_delay

        total_delay_list[0, i] = stem_com_energy.item() + trans_delay.item() + branch_com_delay.item()
        total_energy_list[0, i] = stem_com_energy.item() + trans_energy.item() + branch_com_energy.item()
        last_energy = total_energy_list[0, i]

        total_acc_list[0, i] = acc_curr

        if total_delay_list[0, i] > max_delay:
            delay_vio_num += 1

        acc_curr = util.acc_normalize(acc_curr, curr_context)
        reward_1 = acc_curr
        reward_2 = (max_delay - total_delay_list[0, i]) / max_delay
        reward_3 = total_energy_list[0, i]

        reward = reward_1 + kappa_1 * reward_2 - kappa_2 * reward_3
        # print("reward:", reward, "reward 1:", reward_1, "reward_2:", reward_2, "reward_3", reward_3)
        reward_curr_list.append(reward)

        reward_list[k, i] = reward
    # Averages per episode
    aver_total_delay = np.sum(total_delay_list) / slot_num
    aver_total_energy = np.sum(total_energy_list) / slot_num
    aver_acc = np.sum(total_acc_list) / slot_num
    aver_reward = np.sum(reward_list[k, :]) / slot_num

    print(f'Average Total Delay (s): {aver_total_delay:.2f}')
    print(f'Average Total Energy Consumption (J): {aver_total_energy:.2f}')
    print(f'Remaining Energy Consumption (J): {curr_energy:.2f}')
    print(f'Average Reward: {aver_reward:.2f}')
    print(f'Average Accuracy: {aver_acc:.2f}')
    print(f'Timeout Number: {delay_vio_num}')
    print(f'Retransmission Number: {re_trans_num}')

    eval_total_delay_list[0, k] = aver_total_delay
    eval_total_energy_list[0, k] = aver_total_energy
    eval_reward_list[0, k] = aver_reward
    eval_acc_exp_list[0, k] = aver_acc
    eval_delay_vio_num_list[0, k] = delay_vio_num
    eval_remain_energy_list[0, k] = curr_energy
    eval_re_trans_number_list[0, k] = re_trans_num

mat_name = "eval_data/eval_amac_data.mat"

savemat(mat_name,
        {"amac_eval_episode_total_delay": np.sum(eval_total_delay_list) / len(seed_list),
         "amac_eval_episode_total_energy": np.sum(eval_total_energy_list) / len(seed_list),
         "amac_eval_episode_reward": np.sum(eval_reward_list) / len(seed_list),
         "amac_eval_episode_acc_exp": np.sum(eval_acc_exp_list) / len(seed_list),
         "amac_eval_episode_delay_vio_num": np.sum(eval_delay_vio_num_list) / len(seed_list),
         "amac_eval_episode_remain_energy": np.sum(eval_remain_energy_list) / len(seed_list),
         "amac_eval_episode_re_trans_number": np.sum(eval_re_trans_number_list) / len(seed_list),
         "amac_step_reward_list": reward_list})

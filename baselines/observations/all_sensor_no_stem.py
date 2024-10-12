import numpy as np
import os
import re
import scipy.io as sio
import random
import math
import envs.utils_proposed as util
from scipy.io import savemat
from scipy.io import loadmat
import scipy.special as ss

snr_db_list = np.arange(1, 3.2, 0.2)
data_num = len(snr_db_list)
seed_list = [37]
obs_energy = np.zeros([7,data_num])
obs_delay = np.zeros([7,data_num])
obs_re_trans = np.zeros([7,data_num])

for s in range(len(snr_db_list)):
    slot_num = 1  # Number of time slots
    reward_list = np.zeros([len(seed_list), slot_num])
    sensor_num = 4  # Number of sensors
    bandwidth = 20e6  # System bandwidth (Hz)
    max_power = 1  # Maximum transmit power (W)
    Est_err_para = 0.5
    est_err_para = 0.5
    kappa_1 = 1
    kappa_2 = 1
    enable_re_trans = True
    sub_block_length = 128
    target_snr_db = snr_db_list[s]
    for k in range(len(seed_list)):
        print("seed:", seed_list[k])
        max_energy = 6000  # Maximum energy consumption (J)
        curr_energy = max_energy  # Available energy of current slot
        last_energy = 0  # Consumed energy of last slot

        seed = seed_list[k]
        # Parameter settings
        np.random.seed(seed)

        # Quantized data size
        data_size = np.zeros([1, sensor_num])
        data_size[0, 0] = 6064128
        data_size[0, 1] = data_size[0, 0]
        data_size[0, 2] = 1.8432e6
        data_size[0, 3] = 1.73e6
        # Initialize lists
        total_delay_list = np.zeros([1, slot_num])
        total_energy_list = np.zeros([1, slot_num])
        total_acc_list = np.zeros([1, slot_num])
        acc_vio_num = 0
        delay_vio_num = 0
        re_trans_num = np.nan
        context_flag = 0
        consider_re_trans = 0  # 0 or 1
        max_re_trans_num = 5
        context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
        context_prob = [0.05, 0.05, 0.2, 0.1, 0.2, 0.4]
        curr_context = None
        context_interval = 1  # Interval for context to change
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


        # Data loading and fitting
        # Load TM data
        tm_folder_path = "./system_data/typical modulation"
        tm_pattern = r'snr_([\d.]+)_([\d.]+)_([\d.]+)_(\w+)_esterr_([\d\.]+)_rate_(\d+)_(\d+)'
        tm_mod = "qpsk"  # qpsk, 16qam, 64qam
        tm_coding_rate_num = 1
        tm_coding_rate_den = 2
        tm_coding_rate = tm_coding_rate_num / tm_coding_rate_den
        for filename in os.listdir(tm_folder_path):
            if filename.endswith('.mat'):
                match = re.match(tm_pattern, filename)
                if match:
                    tm_snr_min = float(match.group(1))
                    tm_snr_int = float(match.group(2))
                    tm_snr_max = float(match.group(3))
                    tm_mod_ = match.group(4)
                    tm_est_err_para_ = float(match.group(5))
                    tm_coding_rate_num_ = int(match.group(6))
                    tm_coding_rate_den_ = int(match.group(7))
                    tm_coding_rate_ = tm_coding_rate_num_ / tm_coding_rate_den_

                    if tm_est_err_para_ == est_err_para and tm_mod_ == tm_mod \
                            and tm_coding_rate_ == tm_coding_rate:
                        print("Loading TM SNR-BER data:", filename)
                        tm_file_path = os.path.join(tm_folder_path, filename)
                        tm_data = loadmat(tm_file_path)
                        tm_ber = tm_data['bler']
                        tm_snr_list = np.arange(tm_snr_min, tm_snr_max + tm_snr_int,
                                                     tm_snr_int)

                        # Function fitting
                        tm_degree = 7
                        tm_coefficients = np.polyfit(tm_snr_list.ravel(), tm_ber.ravel(), tm_degree)
                        tm_polynomial_model = np.poly1d(tm_coefficients)

        # Main simulation loop
        for i in range(slot_num):
            print("slot num:", i)
            curr_context_id = context_train_list[context_flag]
            curr_context = context_list[curr_context_id]
            min_acc = util.obtain_min_acc(curr_context)
            if i % context_interval == 0 and i != 0:
                context_flag = context_flag + 1
                curr_context_id = context_train_list[context_flag]
                curr_context = context_list[curr_context_id]
            curr_energy = curr_energy - last_energy
            max_delay = 0.5  # Maximum tolerant delay (s)

            # Calculate SNR
            snr_db = target_snr_db
            snr = 10 ** (snr_db / 10)
            delay_vio_num = 0

            # BER Calculation
            ber_curr = tm_polynomial_model(snr_db)
            ber_curr = np.clip(ber_curr, 0, 1)
            acc_curr = platform_data[curr_context][22, 0]

            # Delay Calculation
            trans_rate = bandwidth * np.log2(1 + snr)
            # print("trans rate:",trans_rate)
            coded_data_size = math.floor(np.sum(data_size) / tm_coding_rate)

            stem_com_delay = np.sum(platform_data["stem_delay"])
            stem_com_energy = np.sum(platform_data["stem_energy"])
            branch_com_delay = platform_data["branch_delay"][22, 0]
            branch_com_energy = platform_data["branch_energy"][22, 0]

            re_trans_delay = 0
            re_trans_energy = 0
            if enable_re_trans:
                re_trans_num = 0
                block_num = np.floor(coded_data_size / sub_block_length)
                tm_per = 1 - (1 - ber_curr) ** sub_block_length
                # print("Using TM")
                print("BER:", ber_curr)
                print("PER:", tm_per)
                print("Block Num:", block_num)
                for j in range(int(block_num)):
                    re_trans_num_block = 0
                    # print("block index:", j)
                    is_trans_success = 0
                    while is_trans_success == 0:
                        # Generate 1 (success) with probability 1-p and 0 (fail) with p
                        is_trans_success = \
                            random.choices([0, 1], weights=[tm_per, 1 - tm_per])[0]
                        if is_trans_success == 1 or re_trans_num_block >= max_re_trans_num:
                            break
                        else:
                            re_trans_num_block = re_trans_num_block + 1
                            tm_per = 1 - (1 - ber_curr) ** (sub_block_length / (1-tm_coding_rate))
                    re_trans_num = re_trans_num + re_trans_num_block
                re_trans_delay = re_trans_num * (
                            (1 / tm_coding_rate - 1) * sub_block_length / trans_rate)
                re_trans_energy = max_power * re_trans_delay
            trans_delay = coded_data_size / trans_rate + re_trans_delay
            trans_energy = max_power * trans_delay + re_trans_energy
            total_delay_list[0, i] = stem_com_energy.item() + trans_delay.item() + branch_com_delay.item()
            total_energy_list[0, i] = stem_com_energy.item() + trans_energy.item() + branch_com_energy.item()
            last_energy = total_energy_list[0, i]
            total_acc_list[0, i] = acc_curr

            if total_delay_list[0, i] > max_delay:
                delay_vio_num += 1
            print("stem com energy:", stem_com_energy.item(), "branch com delay:",branch_com_delay.item(), "trans energy:", trans_energy, "re trans energy:", re_trans_energy)
            if acc_curr < min_acc:
                # print("acc:",acc_exp, "acc_min:",min_acc)
                acc_vio_num = acc_vio_num + 1
            # acc_curr = util.acc_normalize(acc_curr, curr_context)
            reward_1 = 2* ss.erf(acc_curr-min_acc)
            reward_2 = total_delay_list[0, i] / max_delay
            reward_3 = total_energy_list[0, i] / max_energy

            reward = reward_1 - kappa_1 * reward_2 - kappa_2 * reward_3

            reward_list[k, i] = reward[0]
            print(f'Total Delay (s): {total_delay_list[0, i]:.2f}')
            print(f'Total Energy Consumption (J): {total_energy_list[0, i]:.2f}')
            # print(f'Remaining Energy Consumption (J): {curr_energy:.2f}')
            print(f'Reward: {reward[0]:.2f}')
            print(f'Accuracy: {acc_curr:.2f}')
            print(f'Timeout Number: {delay_vio_num}')
            print(f'Retransmission Number: {re_trans_num}')
            print(f'Accuracy violation slot number: {acc_vio_num}')

            obs_energy[6,s] = total_energy_list[0, i]
            obs_delay[6,s] = total_delay_list[0, i]
            obs_re_trans[6,s] = re_trans_num

            mat_name = "baselines/observations/observation.mat"
            savemat(mat_name, {"obs_energy": obs_energy,
                                    "obs_delay": obs_delay,
                               "obs_re_trans": obs_re_trans
                               })
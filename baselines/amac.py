import numpy as np
import os
import re
import scipy.io as sio
import random
import math
import envs.utils_proposed as util
from scipy.io import savemat
import scipy.special as ss
import pandas as pd


def obtain_cqi_and_snr(file_paths, slot_num):
    """
    This function reads multiple CSV files from a list of file paths, extracts "SNR" and "CQI" columns,
    and concatenates them into 1D arrays.

    Parameters:
    - file_paths: list of strings containing paths to directories or specific CSV files.
    - slot_num: number of data points to randomly select from the concatenated data.

    Returns:
    - snr_array: 1D numpy array containing randomly selected "SNR" values.
    - cqi_array: 1D numpy array containing randomly selected "CQI" values.
    """

    # Lists to store SNR and CQI data separately
    snr_list = []
    cqi_list = []

    # Loop through all provided file paths
    for directory_path in file_paths:
        # Check if the path is a directory or a file
        if os.path.isdir(directory_path):
            # If it's a directory, loop through all files in the directory
            for filename in os.listdir(directory_path):
                # Check if the file is a CSV file
                if filename.endswith(".csv"):
                    file_path = os.path.join(directory_path, filename)
                    # Read the CSV file and extract the "SNR" and "CQI" columns
                    df = pd.read_csv(file_path, usecols=["SNR", "CQI"])
                    # Convert non-numeric values to NaN and drop them
                    df["SNR"] = pd.to_numeric(df["SNR"], errors='coerce')
                    df["CQI"] = pd.to_numeric(df["CQI"], errors='coerce')
                    df = df.dropna()  # Drop rows with NaN values
                    # Append the valid "SNR" and "CQI" values to their respective lists
                    snr_list.extend(df["SNR"].values)
                    cqi_list.extend(df["CQI"].values)
        elif os.path.isfile(directory_path) and directory_path.endswith(".csv"):
            # If it's a file, directly read and extract the "SNR" and "CQI" columns
            df = pd.read_csv(directory_path, usecols=["SNR", "CQI"])
            # Convert non-numeric values to NaN and drop them
            df["SNR"] = pd.to_numeric(df["SNR"], errors='coerce')
            df["CQI"] = pd.to_numeric(df["CQI"], errors='coerce')
            df = df.dropna()  # Drop rows with NaN values
            # Append the valid "SNR" and "CQI" values to their respective lists
            snr_list.extend(df["SNR"].values)
            cqi_list.extend(df["CQI"].values)

    # Convert the lists into 1D NumPy arrays
    snr_array = np.array(snr_list)
    cqi_array = np.array(cqi_list)

    # Randomly select `slot_num` values from the arrays
    snr_array = np.random.choice(snr_array, size=slot_num, replace=False)
    cqi_array = np.random.choice(cqi_array, size=slot_num, replace=False)

    return snr_array, cqi_array


class Amac:
    def __init__(self, is_test):
        self.is_test = is_test
        if self.is_test:
            self.seed_list = [40]
            self.slot_num = 3000  # Number of time slots
            self.enable_re_trans = True
        else:
            self.seed_list = [666,555,444,333,111]
            self.slot_num = 10 * 3000  # Number of time slots
            self.enable_re_trans = False

        self.sensor_num = 4  # Number of sensors
        self.bandwidth = 20e6  # System bandwidth (Hz)
        self.max_power = 1  # Maximum transmit power (W)
        self.est_err_para = 0.5
        self.context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
        self.kappa_1 = 2
        self.kappa_2 = 1
        self.kappa_3 = 1
        self.action_freq_list = []  # record the frequency of each action picked
        self.bad_action_freq_list = []  # record the frequency of bad action (acc < acc_min)
        self.sub_block_length = 128
        self.target_snr_db = 1
        self.action_list = []
        # Initialize lists
        self.episode_total_delay_list = np.zeros([1, len(self.seed_list)])
        self.episode_total_energy_list = np.zeros([1, len(self.seed_list)])
        self.episode_reward_list = np.zeros([1, len(self.seed_list)])
        self.episode_acc_exp_list = np.zeros([1, len(self.seed_list)])
        self.episode_delay_vio_num_list = np.zeros([1, len(self.seed_list)])
        self.episode_acc_vio_num_list = np.zeros([1, len(self.seed_list)])
        self.episode_remain_energy_list = np.zeros([1, len(self.seed_list)])
        self.episode_re_trans_num_list = np.zeros([1, len(self.seed_list)])
        self.episode_acc_vio_list = np.zeros([1, len(self.seed_list)])
        self.step_reward_list = np.zeros([len(self.seed_list), self.slot_num])


        self.total_delay_list = np.zeros([1, self.slot_num])
        self.total_energy_list = np.zeros([1, self.slot_num])
        self.total_acc_list = np.zeros([1, self.slot_num])
        self.re_trans_num_list = np.zeros([1, self.slot_num])
        self.acc_vio_list = np.zeros([1, self.slot_num])

        self.acc_vio_num = 0
        self.delay_vio_num = 0
        self.re_trans_num = np.nan
    def run(self):
        for k in range(len(self.seed_list)):
            print("seed:", self.seed_list[k])
            max_energy = 6000  # Maximum energy consumption (J)
            curr_energy = max_energy  # Available energy of current slot
            last_energy = 0  # Consumed energy of last slot
            seed = self.seed_list[k]
            # Parameter settings
            np.random.seed(seed)
            random.seed(seed)

            # Quantized data size
            data_size = np.zeros([1, self.sensor_num])
            for i in range(self.sensor_num):
                data_size[0, i] = np.random.rand() * 10000
            context_flag = 0
            max_re_trans_num = 500

            context_prob = [0.05, 0.05, 0.2, 0.1, 0.2, 0.4]
            curr_context = None
            context_interval = 100  # Interval for context to change
            context_num = int(self.slot_num / context_interval)
            context_train_list = np.random.choice(list(range(len(self.context_list))), size=context_num, p=context_prob)
            # Data loading and fitting
            platform_data = sio.loadmat('system_data/platform_data.mat')

            folder_path = "system_data/typical modulation"
            file_list = os.listdir(folder_path)
            all_valid_file = []

            # Search for valid files
            for file_name in file_list:
                if f"esterr_{self.est_err_para}" in file_name:
                    all_valid_file.append(file_name)

            all_p = []  # Polynomial coefficients
            all_mod_method = []
            all_mod_order = []
            all_rate = []

            wireless_data_paths = ['system_data/5G_dataset/Netflix/Driving/animated-RickandMorty',
                                   'system_data/5G_dataset/Netflix/Driving/Season3-StrangerThings',
                                   'system_data/5G_dataset/Download/Driving',
                                   'system_data/5G_dataset/Amazon_Prime/Driving/Season3-TheExpanse'
                                   ]
            snr_array, cqi_array = obtain_cqi_and_snr(wireless_data_paths, self.slot_num)

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
                    if mod_method == "qpsk":
                        mod_order = 2
                    elif mod_method == "16qam":
                        mod_order = 4
                    elif mod_method == "64qam":
                        mod_order = 6
                    all_mod_order.append(mod_order)
                    all_rate.append(rate)

            # Main simulation loop
            min_acc_list = []
            for i in range(self.slot_num):
                # print("slot num:", i)
                curr_context_id = context_train_list[context_flag]
                curr_context = self.context_list[curr_context_id]
                min_acc = util.obtain_min_acc(curr_context)
                min_acc_list.append(min_acc)
                if i % context_interval == 0 and i != 0:
                    context_flag = context_flag + 1
                    curr_context_id = context_train_list[context_flag]
                    curr_context = self.context_list[curr_context_id]
                curr_energy = curr_energy - last_energy
                max_delay = 0.5  # Maximum tolerant delay (s)

                # Calculate SNR
                # snr_db = float(snr_array[i])
                snr_db = self.target_snr_db
                snr = 10 ** (snr_db / 10)

                # Search for optimal transmission scheme
                reward_curr_list = []
                delay_curr_list = []
                energy_curr_list = []
                re_trans_delay = 0
                re_trans_energy = 0

                for j in range(len(all_valid_file)):
                    p_curr = all_p[j]
                    rate_curr = all_rate[j]
                    # BER Calculation
                    ber_curr = np.polyval(p_curr, snr_db)
                    ber_curr = np.clip(ber_curr, 0, 1)
                    acc_curr = platform_data[curr_context][22, 0] / 100
                    # Delay Calculation
                    trans_rate = all_mod_order[j] * self.bandwidth * np.log2(1 + snr)
                    # print("trans rate:",trans_rate)
                    coded_data_size = math.floor(np.sum(data_size) / rate_curr)

                    trans_delay = coded_data_size / trans_rate
                    trans_energy = trans_delay * self.max_power

                    # Placeholder for stem and branch computation
                    stem_com_delay = np.sum(platform_data["stem_delay"])
                    stem_com_energy = np.sum(platform_data["stem_energy"])
                    branch_com_delay = platform_data["branch_delay"][22, 0]
                    branch_com_energy = platform_data["branch_energy"][22, 0]

                    # Retransmission simulation
                    block_num = 1
                    if self.enable_re_trans:
                        self.re_trans_num = 0
                        block_num = np.floor(coded_data_size / self.sub_block_length)
                        per_curr = 1 - (1 - ber_curr) ** self.sub_block_length
                        # print("Using TM")
                        # print("BER:",tm_ber)
                        # print("PER:", per_curr)
                        for j in range(int(block_num)):
                            re_trans_num_block = 0
                            is_trans_success = 0
                            while is_trans_success == 0:
                                # Generate 1 (success) with probability 1-p and 0 (fail) with p
                                is_trans_success = \
                                    random.choices([0, 1], weights=[per_curr, 1 - per_curr])[0]
                                if is_trans_success == 1 or re_trans_num_block >= max_re_trans_num:
                                    break
                                else:
                                    re_trans_num_block = re_trans_num_block + 1
                                    per_curr = 1 - (1 - ber_curr) ** (self.sub_block_length * (1 - rate_curr))
                            self.re_trans_num = self.re_trans_num + re_trans_num_block
                        re_trans_delay = self.re_trans_num * (
                                self.sub_block_length * (1 - rate_curr)/ trans_rate)
                        re_trans_energy = re_trans_delay * self.max_power
                    trans_delay = trans_delay + re_trans_delay
                    trans_energy = trans_energy + re_trans_energy
                    total_delay = stem_com_delay.item() + trans_delay + branch_com_delay.item()
                    total_energy = stem_com_energy.item() + trans_energy + branch_com_energy.item()
                    last_energy = self.total_energy_list[0, i]
                    reward_1 = ss.erf(acc_curr - min_acc)
                    reward_2 = total_delay / max_delay
                    # reward_3 = self.remain_energy / self.max_energy
                    reward_3 = total_energy / max_energy
                    reward = self.kappa_1 * reward_1 - self.kappa_2 * reward_2 - self.kappa_3 * reward_3

                    reward_curr_list.append(reward)
                    delay_curr_list.append(total_delay)
                    energy_curr_list.append(total_energy)

                # Optimal scheme selection (select the one with the largest reward)
                # print(reward_curr_list)
                opt_index = reward_curr_list.index(np.max(reward_curr_list))
                # print("opt idx:",opt_index)
                # print("self.re_trans_num:",self.re_trans_num)
                self.action_list.append(all_mod_method[opt_index])
                self.step_reward_list[k, i] = np.max(reward_curr_list)
                # print("reward:", self.step_reward_list[k, i])
                self.total_delay_list[0, i] = delay_curr_list[opt_index]
                self.total_energy_list[0, i] = energy_curr_list[opt_index]
                self.re_trans_num_list[0,i] = self.re_trans_num
                self.total_acc_list[0,i] = acc_curr
                if acc_curr <= min_acc:
                    self.acc_vio_num = self.acc_vio_num + 1
                    self.bad_action_freq_list.append(opt_index)
                    self.acc_vio_list[0, i] = np.abs(acc_curr - min_acc)
                self.action_freq_list.append(opt_index)


            # Averages per episode
            aver_total_delay = np.sum(self.total_delay_list) / self.slot_num
            aver_total_energy = np.sum(self.total_energy_list) / self.slot_num
            aver_re_trans_num = np.sum(self.re_trans_num_list) / self.slot_num
            aver_acc = np.sum(self.total_acc_list) / self.slot_num
            aver_reward = np.sum(self.step_reward_list[k, :]) / self.slot_num
            if self.acc_vio_num == 0:
                aver_acc_vio = 0
            else:
                aver_acc_vio = np.sum(self.acc_vio_list) / self.acc_vio_num

            self.acc_vio_num = self.acc_vio_num / self.slot_num

            print(f'Average Total Delay (s): {aver_total_delay:.2f}')
            print(f'Average Total Energy Consumption (J): {aver_total_energy:.2f}')
            print(f'Remaining Energy Consumption (J): {curr_energy:.2f}')
            print(f'Average Reward: {aver_reward:.2f}')
            print(f'Average Accuracy: {aver_acc:.2f}')
            print(f'Timeout Number: {self.delay_vio_num}')
            print(f'Retransmission Number: {aver_re_trans_num}')
            print(f'Accuracy Violation rate: {self.acc_vio_num}')
            print(f'Accuracy Violation: {aver_acc_vio}')
            print(f'min acc: {np.mean(min_acc_list)}')

            self.episode_total_delay_list[0, k] = aver_total_delay
            self.episode_total_energy_list[0, k] = aver_total_energy
            self.episode_reward_list[0, k] = aver_reward
            self.episode_acc_exp_list[0, k] = aver_acc
            self.episode_delay_vio_num_list[0, k] = self.delay_vio_num
            self.episode_remain_energy_list[0, k] = curr_energy
            self.episode_re_trans_num_list[0, k] = aver_re_trans_num
            self.episode_acc_vio_num_list[0, k] = self.acc_vio_num
            self.episode_acc_vio_list[0, k] = aver_acc_vio


if __name__ == '__main__':
    runner = Amac(is_test=False)
    runner.run()

    mat_name = "baselines/amac_data.mat"
    savemat(mat_name,
            {"amac_eval_episode_total_delay": np.sum(runner.episode_total_delay_list) / len(runner.seed_list),
             "amac_eval_episode_total_energy": np.sum(runner.episode_total_energy_list) / len(runner.seed_list),
             "amac_eval_episode_reward": np.sum(runner.episode_reward_list) / len(runner.seed_list),
             "amac_eval_episode_acc_exp": np.sum(runner.episode_acc_exp_list) / len(runner.seed_list),
             "amac_eval_episode_delay_vio_num": np.sum(runner.episode_delay_vio_num_list) / len(runner.seed_list),
             "amac_eval_episode_acc_vio_num": np.sum(runner.episode_acc_vio_num_list) / len(runner.seed_list),
             "amac_eval_episode_remain_energy": np.sum(runner.episode_remain_energy_list) / len(runner.seed_list),
             "amac_eval_episode_re_trans_number": np.sum(runner.episode_re_trans_num_list) / len(runner.seed_list),
             "amac_step_reward_list": runner.step_reward_list
             })

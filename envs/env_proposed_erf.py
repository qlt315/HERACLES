import math
import gym
from gym import spaces, logger
import numpy as np
import os
import re
from scipy.io import loadmat
from tensorboard.compat.tensorflow_stub.dtypes import float32
import envs.utils_proposed as util
import random
import matplotlib.pyplot as plt
import scipy.special as ss


# import envs.mobile_channel_gen

class EnvProposed_erf(gym.Env):
    def __init__(self):
        self.name = "proposed_erf"
        # Parameter settings
        self.slot_num = 3000  # Number of time slots
        self.max_energy = 2000  # Maximum energy consumption (J)
        self.remain_energy = self.max_energy  # Available energy of current slot
        self.last_energy = 0  # Consumed energy of last slot
        self.sensor_num = 4  # Number of sensors
        self.bandwidth = 20e6  # System bandwidth (Hz)
        self.max_power = 1  # Maximum transmit power (W)
        self.est_err_para = 0.5  # Channel estimation error parameter
        self.hm_power_ratio = 0.999  # Choose from 0.5-0.9
        self.data_size = np.zeros([1, self.sensor_num])
        for i in range(self.sensor_num):
            self.data_size[0, i] = np.random.rand(1) * 10000
        self.sub_block_length = 128
        self.step_reward_list = []
        # self.data_size = np.zeros([1, 4])
        # self.data_size[0, 0] = 6064128
        # self.data_size[0, 1] = self.data_size[0, 0]
        # self.data_size[0, 2] = 1.8432e6
        # self.data_size[0, 3] = 1.73e6
        self.target_snr_db = 2
        self.total_delay_list = np.zeros([1, self.slot_num])
        self.total_energy_list = np.zeros([1, self.slot_num])
        self.acc_exp_list = np.zeros([1, self.slot_num])
        self.reward_list = np.zeros([1, self.slot_num])
        self.re_trans_list = np.zeros([1, self.slot_num])

        self.episode_total_delay_list = []
        self.episode_total_energy_list = []
        self.episode_acc_exp_list = []
        self.episode_reward_list = []
        self.episode_delay_vio_num_list = []
        self.episode_remain_energy_list = []
        self.episode_re_trans_num_list = []
        self.episode_acc_vio_num_list = []
        np.seterr(over='ignore')
        self.context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
        # self.context_list = ["sunny", "sunny", "sunny", "sunny", "sunny", "sunny"]
        self.context_prob = [0.05, 0.05, 0.2, 0.1, 0.2, 0.4]
        self.context_interval = 100  # Interval for context to change
        self.context_num = int(self.slot_num / self.context_interval)
        self.context_train_list = np.random.choice(list(range(len(self.context_list))), size=self.context_num,
                                                   p=self.context_prob)
        self.delay_vio_num = 0
        self.context_flag = 0
        self.num_actions = 33
        self.show_fit_plot = False
        self.curr_context = None
        self.enable_re_trans = True  # Consider retransmission or not
        self.re_trans_num = -1
        self.done = False
        # DRL parameter settings
        (self.action_sunny_list, self.action_rain_list, self.action_snow_list,
         self.action_motorway_list, self.action_fog_list, self.action_night_list) = util.action_gen()
        self.action_space = spaces.Discrete(33)
        self.action_freq_list = np.zeros([1,33])  # record the frequency of each action picked
        self.bad_action_freq_list = np.zeros([1, 33]) # record the frequency of bad action (acc < acc_min)
        # Obs: (1) Estimated CQI (1-15) (2) SNR in dB (0-20)   (3) Task context (0-5)  (4) Min accuracy
        obs_low = np.array([1, 0, 0, 0])
        obs_high = np.array([15, 20, 5, 1])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.step_num = 0
        self.episode_num = 0
        self.max_re_trans_num = 200
        self.kappa_1 = 2  # acc reward coefficient
        self.kappa_2 = 1  # delay reward coefficient
        self.kappa_3 = 1  # energy consumption reward coefficient
        self.acc_vio_num = 0
        # Data loading and fitting
        # Load HM data
        self.hm_folder_path = "./system_data/hierarchical modulation/two_layers_data"

        hm_pattern = r'snr_([\d.]+)_([\d.]+)_([\d.]+)_layer1_(\w+)_layer2_(\w+)_esterr_([\d.]+)_rate_(\d+)_(\d+)_power_ratio_(\d+\.?\d*)'
        for filename in os.listdir(self.hm_folder_path):
            if filename.endswith('.mat'):
                match = re.match(hm_pattern, filename)
                if match:
                    self.hm_err_para_ = float(match.group(6))
                    self.hm_power_ratio_ = float(match.group(9))
                    if self.hm_err_para_ != self.est_err_para or self.hm_power_ratio_ != self.hm_power_ratio:
                        continue
                    else:
                        print("Loading 2-Layer HM SNR-BER data:", filename)
                        hm_file_path = os.path.join(self.hm_folder_path, filename)
                        self.hm_snr_min = float(match.group(1))
                        self.hm_snr_int = float(match.group(2))
                        self.hm_snr_max = float(match.group(3))
                        self.hm_layer1_mod = match.group(4)
                        self.hm_layer2_mod = match.group(5)
                        self.hm_coding_rate_num = int(match.group(7))
                        self.hm_coding_rate_den = int(match.group(8))
                        self.hm_coding_rate = self.hm_coding_rate_num / self.hm_coding_rate_den

                        self.hm_data = loadmat(hm_file_path, variable_names=['ber_1', 'ber_2_sic'])
                        self.hm_ber_1 = self.hm_data['ber_1']  # Layer 1 BER
                        self.hm_ber_2 = self.hm_data['ber_2_sic']  # Layer 2 BER with SIC

                        self.hm_ber_1 = self.hm_ber_1[self.hm_ber_1 != 0]
                        self.hm_ber_2 = self.hm_ber_2[self.hm_ber_2 != 0]
                        self.hm_snr_list = np.arange(self.hm_snr_min, self.hm_snr_max + self.hm_snr_int,
                                                     self.hm_snr_int)
                        self.hm_snr_list = self.hm_snr_list[:len(self.hm_ber_1)]
                        # Function fitting
                        self.hm_degree = 5
                        self.hm_coefficients_1 = np.polyfit(self.hm_snr_list.ravel(), self.hm_ber_1.ravel(),
                                                            self.hm_degree)
                        self.hm_coefficients_2 = np.polyfit(self.hm_snr_list.ravel(), self.hm_ber_2.ravel(),
                                                            self.hm_degree)
                        self.hm_polynomial_model_1 = np.poly1d(self.hm_coefficients_1)
                        self.hm_polynomial_model_2 = np.poly1d(self.hm_coefficients_2)

        self.tm_folder_path = "./system_data/typical modulation"
        tm_pattern = r'snr_([\d.]+)_([\d.]+)_([\d.]+)_(\w+)_esterr_([\d\.]+)_rate_(\d+)_(\d+)'
        self.tm_mod = "qpsk"  # qpsk, 16qam, 64qam
        self.tm_coding_rate_num = 1
        self.tm_coding_rate_den = 2
        self.tm_coding_rate = self.tm_coding_rate_num / self.tm_coding_rate_den
        for filename in os.listdir(self.tm_folder_path):
            if filename.endswith('.mat'):
                match = re.match(tm_pattern, filename)
                if match:
                    self.tm_snr_min = float(match.group(1))
                    self.tm_snr_int = float(match.group(2))
                    self.tm_snr_max = float(match.group(3))
                    self.tm_mod_ = match.group(4)
                    self.tm_est_err_para_ = float(match.group(5))
                    self.tm_coding_rate_num_ = int(match.group(6))
                    self.tm_coding_rate_den_ = int(match.group(7))
                    self.tm_coding_rate_ = self.tm_coding_rate_num_ / self.tm_coding_rate_den_

                    if self.tm_est_err_para_ == self.est_err_para and self.tm_mod_ == self.tm_mod \
                            and self.tm_coding_rate_ == self.tm_coding_rate:
                        print("Loading TM SNR-BER data:", filename)
                        tm_file_path = os.path.join(self.tm_folder_path, filename)
                        self.tm_data = loadmat(tm_file_path)
                        self.tm_ber = self.tm_data['bler']
                        self.tm_snr_list = np.arange(self.tm_snr_min, self.tm_snr_max + self.tm_snr_int,
                                                     self.tm_snr_int)

                        # Function fitting
                        self.tm_degree = 7
                        self.tm_coefficients = np.polyfit(self.tm_snr_list.ravel(), self.tm_ber.ravel(), self.tm_degree)
                        self.tm_polynomial_model = np.poly1d(self.tm_coefficients)

        wireless_data_path = 'system_data/5G_dataset/Netflix/Driving/animated-RickandMorty'
        self.snr_array, self.cqi_array = util.obtain_cqi_and_snr(wireless_data_path, self.slot_num)

    def step(self, action):
        # print("Episode index:", self.episode_num, "Step index:", self.step_num)
        max_delay = np.random.uniform(low=0.3, high=1, size=1)
        curr_context_id = self.context_train_list[self.context_flag]
        self.curr_context = self.context_list[curr_context_id]
        if self.step_num % self.context_interval == 0 and self.step_num != 0:
            self.context_flag = self.context_flag + 1
            curr_context_id = self.context_train_list[self.context_flag]
            self.curr_context = self.context_list[curr_context_id]
        min_acc = util.obtain_min_acc(self.curr_context)
        # h = (np.random.randn(1) + 1j  # Wireless channel
        #      * np.random.randn(1)) / np.sqrt(2)
        # h = mobile_channel_gen.calculate_channel_gain(self.step_num)

        # Channel estimation (in CQI)
        cqi = int(self.cqi_array[self.step_num])
        cqi_est = util.estimate_cqi(cqi, self.est_err_para)

        action_info = util.action_mapping(self.action_sunny_list, self.action_rain_list, self.action_snow_list,
                                          self.action_motorway_list, self.action_fog_list, self.action_night_list,
                                          self.curr_context, action)
        # Calculate SNR and trans rate
        # tm_snr_db = float(self.snr_array[self.step_num])
        tm_snr_db = self.target_snr_db
        tm_snr_db = np.clip(tm_snr_db, np.min(self.tm_snr_list), np.max(self.tm_snr_list))
        # print("SNR (dB):", tm_snr_db)
        tm_snr = 10 ** (tm_snr_db / 10)

        tm_ber = np.clip(self.tm_polynomial_model(tm_snr_db), 0.00001, 0.99999)
        tm_trans_rate = self.bandwidth * np.log2(1 + tm_snr)  # Bit / s

        hm_snr_1 = tm_snr * self.hm_power_ratio
        hm_snr_2 = tm_snr * (1 - self.hm_power_ratio)

        hm_snr_1_db = 10 * np.log10(hm_snr_1)
        hm_snr_2_db = 10 * np.log10(hm_snr_2)
        hm_snr_1_db = np.clip(hm_snr_1_db, np.min(self.hm_snr_list), np.max(self.hm_snr_list))
        hm_snr_2_db = np.clip(hm_snr_2_db, np.min(self.hm_snr_list), np.max(self.hm_snr_list))
        hm_ber_1 = np.clip(self.hm_polynomial_model_1(hm_snr_1_db), 0.00001, 0.99999)
        hm_ber_2 = np.clip(self.hm_polynomial_model_2(hm_snr_2_db), 0.00001, 0.99999)

        hm_trans_rate = tm_trans_rate

        # Calculate PER of each branch (totally 21 branches)
        # print(tm_ber, hm_ber_1, hm_ber_2)
        # per_list = util.per_list_gen(tm_ber, hm_ber_1, hm_ber_2, self.data_size, self.tm_coding_rate,
        #                              self.hm_coding_rate)
        re_trans_energy = 0
        # Calculate delay
        re_trans_delay = 0
        if len(action_info.fusion_name) == 1:  # TM

            data_size_idx = action_info.fusion_name[0] - 1
            data_size = self.data_size[0, data_size_idx] / self.tm_coding_rate

            # Retransmission simulation
            if self.enable_re_trans:
                self.re_trans_num = 0
                block_num = np.floor(data_size / self.sub_block_length)
                tm_per = 1 - (1 - tm_ber) ** self.sub_block_length
                # print("Using TM")
                # print("PER:", tm_per)
                # print("Block num:", block_num)
                for j in range(int(block_num)):
                    re_trans_num_block = 0
                    is_trans_success = 0
                    while is_trans_success == 0:
                        # Generate 1 (success) with probability 1-p and 0 (fail) with p
                        is_trans_success = \
                            random.choices([0, 1], weights=[tm_per, 1 - tm_per])[0]
                        if is_trans_success == 1 or re_trans_num_block >= self.max_re_trans_num:
                            break
                        else:
                            re_trans_num_block = re_trans_num_block + 1
                            tm_per = 1 - (1 - tm_ber) ** (self.sub_block_length / (1 - self.tm_coding_rate))
                    self.re_trans_num = self.re_trans_num + re_trans_num_block
                re_trans_delay = self.re_trans_num * (
                            (1 / self.tm_coding_rate - 1) * self.sub_block_length / tm_trans_rate)
                re_trans_energy = self.max_power * re_trans_delay

            trans_delay = data_size / tm_trans_rate + re_trans_delay
        else:  # HM
            data_size_idx = [i - 1 for i in action_info.fusion_name]
            data_size = np.max(self.data_size[0, data_size_idx]) / self.hm_coding_rate

            # Retransmission simulation
            if self.enable_re_trans:
                order = action_info.fusion_name
                hm_per_1 = 1 - (1 - hm_ber_1) ** self.sub_block_length
                hm_per_2 = 1 - (1 - hm_ber_2) ** self.sub_block_length
                hm_per = 1 - (1 - hm_per_1) * (1 - hm_per_2)
                block_num_1 = np.floor(self.data_size[0, order[0] - 1] / self.hm_coding_rate / self.sub_block_length)
                block_num_2 = np.floor(self.data_size[0, order[1] - 1] / self.hm_coding_rate / self.sub_block_length)
                for j in range(int(max(block_num_1, block_num_2))):
                    re_trans_num_block = 0
                    is_trans_success = 0
                    while is_trans_success == 0:
                        # Generate 1 (success) with probability 1-p and 0 (fail) with p
                        is_trans_success = \
                            random.choices([0, 1], weights=[hm_per, 1 - hm_per])[0]
                        if is_trans_success == 1 or re_trans_num_block >= self.max_re_trans_num:
                            break
                        else:
                            re_trans_num_block = re_trans_num_block + 1
                            hm_per_1 = 1 - (1 - hm_ber_1) ** (self.sub_block_length / (1 - self.tm_coding_rate))
                            hm_per_2 = 1 - (1 - hm_ber_2) ** (self.sub_block_length / (1 - self.tm_coding_rate))
                            hm_per = 1 - (1 - hm_per_1) * (1 - hm_per_2)
                    self.re_trans_num = self.re_trans_num + re_trans_num_block
                re_trans_delay = self.re_trans_num * (
                            (1 / self.hm_coding_rate - 1) * self.sub_block_length / hm_trans_rate)
                re_trans_energy = self.max_power * re_trans_delay
                # print("re trans delay:",re_trans_delay)
            trans_delay = data_size / hm_trans_rate + re_trans_delay
        # print("Re-trans number:", self.re_trans_num)
        # print(re_trans_delay,re_trans_energy)
        com_delay = action_info.com_delay
        total_delay = trans_delay + com_delay
        self.total_delay_list[0, self.step_num] = total_delay
        self.re_trans_list[0, self.step_num] = self.re_trans_num
        # Calculate energy consumption
        trans_energy = self.max_power * trans_delay + re_trans_energy
        com_energy = action_info.com_energy
        total_energy = trans_energy + com_energy
        self.total_energy_list[0, self.step_num] = total_energy
        self.remain_energy = self.remain_energy - total_energy

        # Reward calculation
        # # # Option 1: Calculate accuracy (expectation)
        # acc_exp = util.acc_exp_gen(per_list, self.curr_context)
        # # Normalize the reward (Acc reward should be further normalized)
        # acc_exp = util.acc_normalize(acc_exp, self.curr_context)

        # Option 2: Don't consider expectation
        acc_exp = action_info.acc / 100
        if acc_exp < min_acc:
            # print("acc:",acc_exp, "acc_min:",min_acc)
            self.acc_vio_num = self.acc_vio_num + 1
            self.bad_action_freq_list[0,action] +=1
        # acc_exp = util.acc_normalize(acc_exp, self.curr_context)

        self.acc_exp_list[0, self.step_num] = acc_exp
        # Reward calculation
        reward_1 = ss.erf(acc_exp - min_acc)
        reward_2 = total_delay / max_delay
        reward_3 = total_energy / self.max_energy
        # reward_3 = total_energy
        reward = self.kappa_1 * reward_1 - self.kappa_2 * reward_2 - self.kappa_3 * reward_3
        # print(reward_1, reward_2, reward_3, reward)
        # reward = acc_exp + self.kappa_1 * (max_delay - total_delay) + self.kappa_2 * self.remain_energy

        # print("slot index:", self.step_num, "action info:", action_info.fusion_name, "branch:", action_info.backbone,
        #       "reward:", reward, "reward 1:", reward_1, "reward_2:", reward_2, "reward_3", -reward_3, "context:",
        #       self.curr_context)
        self.reward_list[0, self.step_num] = reward
        self.step_reward_list.append(reward.item())
        # State calculation
        state = [cqi_est, tm_snr, curr_context_id, min_acc.item()]

        if total_delay > max_delay:
            self.delay_vio_num = self.delay_vio_num + 1

        if self.step_num >= self.slot_num - 1:
            print("Episode index:", self.episode_num)
            self.episode_num = self.episode_num + 1
            self.done = True

            self.acc_vio_num = self.acc_vio_num / self.slot_num

            episode_total_delay = np.sum(self.total_delay_list) / self.slot_num
            episode_total_energy = np.sum(self.total_energy_list) / self.slot_num
            episode_acc_exp = np.sum(self.acc_exp_list) / self.slot_num
            episode_reward = np.sum(self.reward_list) / self.slot_num
            episode_re_trans_num = np.sum(self.re_trans_list) / self.slot_num

            print("Average total delay (s) of current episode:", episode_total_delay)
            print("Average total energy consumption (J)", episode_total_energy, "Remain energy (J)", self.remain_energy)
            print("Average accuracy expectation", episode_acc_exp)
            print("Average episode reward", episode_reward)
            print("Delay violation slot number:", self.delay_vio_num)
            print("Retransmission number:", episode_re_trans_num)
            print("Accuracy violation rate:", self.acc_vio_num)

            self.episode_total_delay_list.append(episode_total_delay)
            self.episode_total_energy_list.append(episode_total_energy)
            self.episode_acc_exp_list.append(episode_acc_exp)
            self.episode_reward_list.append(episode_reward)
            self.episode_delay_vio_num_list.append(self.delay_vio_num)
            self.episode_remain_energy_list.append(self.remain_energy.item())
            self.episode_re_trans_num_list.append(episode_re_trans_num)
            self.episode_acc_vio_num_list.append(self.acc_vio_num)

        self.step_num = self.step_num + 1
        self.action_freq_list[0, action] +=1

        return np.array(state), reward, self.done

    def reset(self):
        cqi_init = 1
        snr_init = 5
        context_id_init = 1
        min_acc_init = 0.1
        state_init = [cqi_init, snr_init, context_id_init, min_acc_init]
        self.context_flag = 0
        self.step_num = 0
        self.delay_vio_num = 0
        self.acc_vio_num = 0
        self.remain_energy = self.max_energy  # Available energy of current slot
        self.done = False
        self.total_delay_list = np.zeros([1, self.slot_num])
        self.total_energy_list = np.zeros([1, self.slot_num])
        self.acc_exp_list = np.zeros([1, self.slot_num])
        self.reward_list = np.zeros([1, self.slot_num])
        self.re_trans_list = np.zeros([1, self.slot_num])

        self.episode_total_delay_list = []
        self.episode_total_energy_list = []
        self.episode_acc_exp_list = []
        self.episode_reward_list = []
        self.episode_delay_vio_num_list = []
        self.episode_remain_energy_list = []
        self.episode_re_trans_num_list = []
        self.episode_acc_vio_num_list = []
        return np.array(state_init)

    def input_est_err(self, est_err_para):
        self.est_err_para = est_err_para
        self.hm_folder_path = "./system_data/hierarchical modulation/two_layers_data"

        hm_pattern = r'snr_([\d.]+)_([\d.]+)_([\d.]+)_layer1_(\w+)_layer2_(\w+)_esterr_([\d.]+)_rate_(\d+)_(\d+)_power_ratio_(\d+\.?\d*)'
        for filename in os.listdir(self.hm_folder_path):
            if filename.endswith('.mat'):
                match = re.match(hm_pattern, filename)
                if match:
                    self.hm_err_para_ = float(match.group(6))
                    self.hm_power_ratio_ = float(match.group(9))
                    if self.hm_err_para_ != self.est_err_para or self.hm_power_ratio_ != self.hm_power_ratio:
                        continue
                    else:
                        print("Loading 2-Layer HM SNR-BER data:", filename)
                        hm_file_path = os.path.join(self.hm_folder_path, filename)
                        self.hm_snr_min = float(match.group(1))
                        self.hm_snr_int = float(match.group(2))
                        self.hm_snr_max = float(match.group(3))
                        self.hm_layer1_mod = match.group(4)
                        self.hm_layer2_mod = match.group(5)
                        self.hm_coding_rate_num = int(match.group(7))
                        self.hm_coding_rate_den = int(match.group(8))
                        self.hm_coding_rate = self.hm_coding_rate_num / self.hm_coding_rate_den

                        self.hm_data = loadmat(hm_file_path, variable_names=['ber_1', 'ber_2_sic'])
                        self.hm_ber_1 = self.hm_data['ber_1']  # Layer 1 BER
                        self.hm_ber_2 = self.hm_data['ber_2_sic']  # Layer 2 BER with SIC

                        self.hm_ber_1 = self.hm_ber_1[self.hm_ber_1 != 0]
                        self.hm_ber_2 = self.hm_ber_2[self.hm_ber_2 != 0]
                        self.hm_snr_list = np.arange(self.hm_snr_min, self.hm_snr_max + self.hm_snr_int,
                                                     self.hm_snr_int)
                        self.hm_snr_list = self.hm_snr_list[:len(self.hm_ber_1)]
                        # Function fitting
                        self.hm_degree = 5
                        self.hm_coefficients_1 = np.polyfit(self.hm_snr_list.ravel(), self.hm_ber_1.ravel(),
                                                            self.hm_degree)
                        self.hm_coefficients_2 = np.polyfit(self.hm_snr_list.ravel(), self.hm_ber_2.ravel(),
                                                            self.hm_degree)
                        self.hm_polynomial_model_1 = np.poly1d(self.hm_coefficients_1)
                        self.hm_polynomial_model_2 = np.poly1d(self.hm_coefficients_2)

        self.tm_folder_path = "./system_data/typical modulation"
        tm_pattern = r'snr_([\d.]+)_([\d.]+)_([\d.]+)_(\w+)_esterr_([\d\.]+)_rate_(\d+)_(\d+)'
        self.tm_mod = "qpsk"  # qpsk, 16qam, 64qam
        self.tm_coding_rate_num = 1
        self.tm_coding_rate_den = 2
        self.tm_coding_rate = self.tm_coding_rate_num / self.tm_coding_rate_den
        for filename in os.listdir(self.tm_folder_path):
            if filename.endswith('.mat'):
                match = re.match(tm_pattern, filename)
                if match:
                    self.tm_snr_min = float(match.group(1))
                    self.tm_snr_int = float(match.group(2))
                    self.tm_snr_max = float(match.group(3))
                    self.tm_mod_ = match.group(4)
                    self.tm_est_err_para_ = float(match.group(5))
                    self.tm_coding_rate_num_ = int(match.group(6))
                    self.tm_coding_rate_den_ = int(match.group(7))
                    self.tm_coding_rate_ = self.tm_coding_rate_num_ / self.tm_coding_rate_den_

                    if self.tm_est_err_para_ == self.est_err_para and self.tm_mod_ == self.tm_mod \
                            and self.tm_coding_rate_ == self.tm_coding_rate:
                        print("Loading TM SNR-BER data:", filename)
                        tm_file_path = os.path.join(self.tm_folder_path, filename)
                        self.tm_data = loadmat(tm_file_path)
                        self.tm_ber = self.tm_data['bler']
                        self.tm_snr_list = np.arange(self.tm_snr_min, self.tm_snr_max + self.tm_snr_int,
                                                     self.tm_snr_int)

                        # Function fitting
                        self.tm_degree = 7
                        self.tm_coefficients = np.polyfit(self.tm_snr_list.ravel(), self.tm_ber.ravel(), self.tm_degree)
                        self.tm_polynomial_model = np.poly1d(self.tm_coefficients)

    def input_snr(self, snr_db):
        self.target_snr_db = snr_db


# this ablation scheme uses the typical modulation method,
# i.e., all $K$ selected semantic features are transmitted
# without considering the different priorities.

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import os
import re
from scipy.io import loadmat
import envs.utils_tem as util
import random
import matplotlib.pyplot as plt


class EnvTEM(gym.Env):
    def __init__(self):
        self.name = "tem"
        self.seed()
        # Parameter settings
        self.slot_num = 3000  # Number of time slots
        self.max_energy = 2000  # Maximum energy consumption (J)
        self.remain_energy = self.max_energy  # Available energy of current slot
        self.last_energy = 0  # Consumed energy of last slot
        self.sensor_num = 4  # Number of sensors
        self.bandwidth = 20e6  # System bandwidth (Hz)
        self.max_power = 1  # Maximum transmit power (W)
        self.est_err_para = 0.5  # Channel estimation error parameter
        self.hm_power_ratio = 0.9  # Choose from 0.5-0.9
        self.data_size = np.zeros([1, self.sensor_num])
        for i in range(self.sensor_num):
            self.data_size[0, i] = np.random.rand(1) * 5000
        self.total_delay_list = np.zeros([1, self.slot_num])
        self.total_energy_list = np.zeros([1, self.slot_num])
        self.acc_exp_list = np.zeros([1, self.slot_num])
        self.reward_list = np.zeros([1, self.slot_num])
        self.step_reward_list = []
        self.episode_total_delay_list = []
        self.episode_total_energy_list = []
        self.episode_acc_exp_list = []
        self.episode_reward_list = []
        self.episode_delay_vio_num_list = []
        self.episode_remain_energy_list = []
        self.episode_re_trans_num_list = []
        np.seterr(over='ignore')
        self.context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
        self.context_prob = [0.05, 0.05, 0.2, 0.1, 0.2, 0.4]
        self.context_interval = 100  # Interval for context to change
        self.context_num = int(self.slot_num / self.context_interval)
        self.context_train_list = np.random.choice(list(range(len(self.context_list))),
                                                   size=self.context_num, p=self.context_prob)
        self.delay_vio_num = 0
        self.context_flag = 0
        self.target_snr_db = 10
        self.show_fit_plot = False
        self.curr_context = None
        self.enable_re_trans = True  # Consider retransmission or not
        self.re_trans_num = -1
        self.sub_block_length = 128
        # DRL parameter settings
        (self.action_sunny_list, self.action_rain_list, self.action_snow_list,
         self.action_motorway_list, self.action_fog_list, self.action_night_list) = util.action_gen()
        self.action_space = spaces.Discrete(21)

        # Obs: (1) Estimated CQI (1-15) (2) SNR in dB (0-20)   (3) Task context (0-5) (4) Remaining energy (5) Maximum tolerant delay
        obs_low = np.array([1, 0, 0, 0, 0])
        obs_high = np.array([15, 20, 5, self.max_energy, 1])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.step_num = 0
        self.episode_num = 0
        self.max_re_trans_num = 5000
        self.kappa_1 = 0.5  # delay reward coefficient
        self.kappa_2 = 0.5  # energy consumption reward coefficient

        # Data loading and fitting
        # Load TM data
        self.tm_folder_path = "./system_data/typical modulation"
        tm_pattern = r'snr_([\d.]+)_([\d.]+)_([\d.]+)_(\w+)_esterr_([\d\.]+)_rate_(\d+)_(\d+)'
        self.tm_mod = "16qam"  # qpsk, 16qam, 64qam
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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # print("Episode index:", self.episode_num, "Step index:", self.step_num)
        max_delay = np.random.uniform(low=0.3, high=1, size=1)
        curr_context_id = self.context_train_list[self.context_flag]
        self.curr_context = self.context_list[curr_context_id]
        if self.step_num % self.context_interval == 0 and self.step_num != 0:
            self.context_flag = self.context_flag + 1
            curr_context_id = self.context_train_list[self.context_flag]
            self.curr_context = self.context_list[curr_context_id]
        # h = (np.random.randn(1) + 1j  # Wireless channel
        #      * np.random.randn(1)) / np.sqrt(2)

        # # Channel estimation
        # est_err = self.est_err_para * np.abs(h)
        # h_est = h + est_err + est_err * 1j
        # noise_power = abs(h_est) * self.max_power / (10 ** (self.target_snr_db / 10))
        # Channel estimation (in CQI)

        cqi = int(self.cqi_array[self.step_num])
        cqi_est = util.estimate_cqi(cqi, self.est_err_para)

        action_info = util.action_mapping(self.action_sunny_list, self.action_rain_list, self.action_snow_list,
                                          self.action_motorway_list, self.action_fog_list, self.action_night_list,
                                          self.curr_context, action)

        # Calculate SNR and trans rate
        tm_snr_db = float(self.snr_array[self.step_num])
        tm_snr_db = np.clip(tm_snr_db, np.min(self.tm_snr_list), np.max(self.tm_snr_list))
        # print("SNR (dB):", tm_snr_db)
        tm_snr =  10 ** (tm_snr_db / 10)

        tm_ber = np.clip(self.tm_polynomial_model(tm_snr_db), 0.00001, 0.99999)
        tm_trans_rate = self.bandwidth * np.log2(1 + tm_snr)  # Bit / s

        # Calculate PER of each branch (totally 21 branches)
        # print(tm_ber, hm_ber_1, hm_ber_2)
        # per_list = util.per_list_gen(tm_ber, self.data_size, self.tm_coding_rate)

        # Calculate delay
        re_trans_delay = 0
        data_size_idx = action_info.fusion_name[0] - 1
        data_size = self.data_size[0, data_size_idx] / self.tm_coding_rate

        # Retransmission simulation
        if self.enable_re_trans:
            self.re_trans_num = 0
            block_num = np.floor(data_size / self.sub_block_length)
            tm_per = 1 - (1 - tm_ber) ** self.sub_block_length
            # print("Using TM")
            # print("BER:",tm_ber)
            # print("PER:", per_curr)
            for j in range(int(block_num)):
                is_trans_success = 0
                while is_trans_success == 0:
                    # Generate 1 (success) with probability 1-p and 0 (fail) with p
                    is_trans_success = \
                        random.choices([0, 1], weights=[tm_per, 1 - tm_per])[0]
                    if is_trans_success == 1 or self.re_trans_num >= self.max_re_trans_num:
                        break
                    else:
                        self.re_trans_num = self.re_trans_num + 1
            re_trans_delay = self.re_trans_num * (
                        (1 / self.tm_coding_rate - 1) * self.sub_block_length / tm_trans_rate)
        trans_delay = data_size / tm_trans_rate + re_trans_delay

        # print("Re-trans number:", self.re_trans_num)
        com_delay = action_info.com_delay
        total_delay = trans_delay + com_delay
        self.total_delay_list[0, self.step_num] = total_delay

        # Calculate energy consumption
        trans_energy = self.max_power * trans_delay
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
        # acc_exp = util.acc_normalize(acc_exp, self.curr_context)

        reward_1 = acc_exp
        reward_2 = (max_delay - total_delay) / max_delay
        # reward_3 = self.remain_energy / self.max_energy
        reward_3 = total_energy
        reward = reward_1 + self.kappa_1 * reward_2 - self.kappa_2 * reward_3
        # reward = acc_exp + self.kappa_1 * (max_delay - total_delay) + self.kappa_2 * self.remain_energy
        self.reward_list[0, self.step_num] = reward
        self.step_reward_list.append(reward.item())


        # State calculation
        state = [cqi_est, tm_snr, curr_context_id, self.remain_energy.item(), max_delay.item()]
        # print("action info:", action_info.fusion_name, "reward:", reward, "reward 1:", reward_1, "reward_2:", reward_2,
        #       "reward_3", reward_3)

        if total_delay > max_delay:
            self.delay_vio_num = self.delay_vio_num + 1

        if self.step_num >= self.slot_num - 1:
            print("Episode index:", self.episode_num)
            self.episode_num = self.episode_num + 1
            done = True

            episode_total_delay = np.sum(self.total_delay_list) / self.slot_num
            episode_total_energy = np.sum(self.total_energy_list) / self.slot_num
            episode_acc_exp = np.sum(self.acc_exp_list) / self.slot_num
            episode_reward = np.sum(self.reward_list) / self.slot_num

            print("Average total delay (s) of current episode:", episode_total_delay)
            print("Average total energy consumption (J)", episode_total_energy, "Remain energy (J)", self.remain_energy)
            print("Average accuracy expectation", episode_acc_exp)
            print("Average episode reward", episode_reward)
            print("Delay violation slot number:", self.delay_vio_num)
            print("Retransmission number:", self.re_trans_num)

            self.episode_total_delay_list.append(episode_total_delay)
            self.episode_total_energy_list.append(episode_total_energy)
            self.episode_acc_exp_list.append(episode_acc_exp)
            self.episode_reward_list.append(episode_reward)
            self.episode_delay_vio_num_list.append(self.delay_vio_num)
            self.episode_remain_energy_list.append(self.remain_energy)
            self.episode_re_trans_num_list.append(self.re_trans_num)

        else:
            done = False

        self.step_num = self.step_num + 1

        return np.array(state), reward, done

    def reset(self):
        cqi_init = 1
        snr_init = 5
        context_id_init = 1
        max_delay_init = 0.3
        state_init = [cqi_init, snr_init, context_id_init, self.max_energy, max_delay_init]
        self.context_flag = 0
        self.step_num = 0
        self.delay_vio_num = 0
        self.remain_energy = self.max_energy  # Available energy of current slot
        self.done = False
        return np.array(state_init)

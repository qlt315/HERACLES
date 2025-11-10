# This ablation scheme applies a static semantic encoder where
# all the stems are activated and all the semantic features are
# transmitted based on the hierarchical modulation.

import random
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import os
import re
from scipy.io import loadmat
import HERACLES.envs.utils_sse as util
import matplotlib.pyplot as plt
import scipy.special as ss
import pandas as pd

class EnvSSE(gym.Env):
    def __init__(self):
        self.name = "sse"
        # Parameter settings
        self.slot_num = 3000  # Number of time slots
        self.max_energy = 2000  # Maximum energy consumption (J)
        self.remain_energy = self.max_energy  # Available energy of current slot
        self.last_energy = 0  # Consumed energy of last slot
        self.sensor_num = 4  # Number of sensors
        self.bandwidth = 20e6  # System bandwidth (Hz)
        self.max_power = 1  # Maximum transmit power (W)
        self.est_err_para = 0.5  # Channel estimation error parameter
        self.data_size = np.zeros([1, self.sensor_num])
        for i in range(self.sensor_num):
            self.data_size[0, i] = np.random.rand(1) * 10000
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
        self.acc_vio_list = np.zeros([1, self.slot_num])
        self.step_reward_list = []
        self.episode_total_delay_list = []
        self.episode_total_energy_list = []
        self.episode_acc_exp_list = []
        self.episode_reward_list = []
        self.episode_delay_vio_num_list = []
        self.episode_remain_energy_list = []
        self.episode_re_trans_num_list = []
        self.episode_acc_vio_num_list = []
        self.episode_acc_vio_list = []
        self.context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
        # self.context_list = ["sunny", "sunny", "sunny", "sunny", "sunny", "sunny"]
        self.context_prob = [0.05, 0.05, 0.2, 0.1, 0.2, 0.4]
        self.context_interval = 100  # Interval for context to change
        self.context_num = int(self.slot_num / self.context_interval)
        self.context_train_list = np.random.choice(list(range(len(self.context_list))),size=self.context_num, p=self.context_prob)
        self.context_flag = 0
        self.delay_vio_num = 0
        self.acc_vio_num = 0
        self.curr_context = None
        self.show_fit_plot = False
        self.re_trans_num = -1
        self.enable_re_trans = True  # Consider retransmission or not
        self.sub_block_length = 128
        # DRL parameter settings
        (self.action_sunny_list, self.action_rain_list, self.action_snow_list,
         self.action_motorway_list, self.action_fog_list, self.action_night_list) = util.action_gen()
        self.action_space = spaces.Discrete(24)
        self.action_freq_list = np.zeros([1,24])  # record the frequency of each action picked
        self.bad_action_freq_list = np.zeros([1, 24]) # record the frequency of bad action (acc < acc_min)
        # Obs: (1) Estimated CQI (1-15) (2) SNR in dB (0-20)   (3) Task context (0-5)  (4) Min accuracy
        obs_low = np.array([1, 0, 0, 0])
        obs_high = np.array([15, 20, 5, 1])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.step_num = 0
        self.episode_num = 0
        self.max_re_trans_num = 500
        self.kappa_1 = 2  # acc reward coefficient
        self.kappa_2 = 1  # delay reward coefficient
        self.kappa_3 = 1  # energy consumption reward coefficient

        # Data loading and fitting
        self.mcs_df = pd.read_csv("/home/ababu/mcs_performance_table.csv")

# Fit polynomial BER models and store spectral efficiencies
        self.mcs_models = {}
        self.mcs_efficiencies = {}

        for mcs_index, group in self.mcs_df.groupby("MCS_Index"):
            sinr = group["SINR_dB"].values
            ber = group["BER"].values
            eff = group["Spectral_Efficiency_bpsHz"].values[0]  # constant per MCS index
            coeffs = np.polyfit(sinr, ber, deg=5)
            self.mcs_models[mcs_index] = np.poly1d(coeffs)
            self.mcs_efficiencies[mcs_index] = eff
        wireless_data_path = '/home/ababu/HERACLES/system_data/5G_dataset/Netflix/Driving/animated-RickandMorty'
        self.snr_array, self.cqi_array = util.obtain_cqi_and_snr(wireless_data_path, self.slot_num)
    def step(self, action):
        max_delay = np.random.uniform(low=0.3, high=1, size=1)
        curr_context_id = self.context_train_list[self.context_flag]
        self.curr_context = self.context_list[curr_context_id]
        if self.step_num % self.context_interval == 0 and self.step_num != 0:
            self.context_flag += 1
            curr_context_id = self.context_train_list[self.context_flag]
            self.curr_context = self.context_list[curr_context_id]

        min_acc = util.obtain_min_acc(self.curr_context)
        cqi = int(self.cqi_array[self.step_num])
        cqi_est = util.estimate_cqi(cqi, self.est_err_para)

        action_info = util.action_mapping(self.action_sunny_list, self.action_rain_list, self.action_snow_list,
                                          self.action_motorway_list, self.action_fog_list, self.action_night_list,
                                          self.curr_context, action)

        mcs_index = action
        snr_db = self.target_snr_db
        snr_db = np.clip(snr_db, self.mcs_df["SINR_dB"].min(), self.mcs_df["SINR_dB"].max())
        snr_linear = 10 ** (snr_db / 10)

        ber_model = self.mcs_models.get(mcs_index, None)
        ber = np.clip(ber_model(snr_db), 0.00001, 0.99999) if ber_model else 0.5
        spectral_eff = self.mcs_efficiencies.get(mcs_index, 0.5)
        trans_rate = spectral_eff * self.bandwidth

        order = action_info.fusion_name
        data_size = np.max(self.data_size[0, [i - 1 for i in order]])
        block_num = np.floor(data_size / self.sub_block_length)

        trans_delay = data_size / trans_rate
        re_trans_delay = 0
        re_trans_energy = 0

        if self.enable_re_trans:
            self.re_trans_num = 0
            per = 1 - ((1 - ber) ** self.sub_block_length)
            for _ in range(int(block_num)):
                re_trans_num_block = 0
                is_trans_success = 0
                while is_trans_success == 0:
                    is_trans_success = random.choices([0, 1], weights=[per, 1 - per])[0]
                    if is_trans_success == 1 or re_trans_num_block >= self.max_re_trans_num:
                        break
                    else:
                        re_trans_num_block += 1
                        per = 1 - ((1 - ber) ** (self.sub_block_length / (1 - spectral_eff / np.log2(1 + snr_linear))))
                self.re_trans_num += re_trans_num_block
            re_trans_delay = self.re_trans_num * ((1 / spectral_eff - 1) * self.sub_block_length / trans_rate)
            re_trans_energy = self.max_power * re_trans_delay

        trans_delay += re_trans_delay
        com_delay = action_info.com_delay
        total_delay = trans_delay + com_delay
        self.total_delay_list[0, self.step_num] = total_delay
        self.re_trans_list[0, self.step_num] = self.re_trans_num

        trans_energy = self.max_power * trans_delay + re_trans_energy
        com_energy = action_info.com_energy
        total_energy = trans_energy + com_energy
        self.total_energy_list[0, self.step_num] = total_energy
        self.remain_energy -= total_energy

        acc_exp = action_info.acc / 100
        self.acc_exp_list[0, self.step_num] = acc_exp
        if acc_exp < min_acc:
            self.acc_vio_num += 1
            self.bad_action_freq_list[0, action] += 1
            self.acc_vio_list[0, self.step_num] = np.abs(acc_exp - min_acc)

        self.action_freq_list[0, action] += 1

        reward_1 = ss.erf(acc_exp - min_acc)
        reward_2 = total_delay / max_delay
        reward_3 = total_energy / self.max_energy
        reward = self.kappa_1 * reward_1 - self.kappa_2 * reward_2 - self.kappa_3 * reward_3
        self.reward_list[0, self.step_num] = reward
        self.step_reward_list.append(reward.item())

        state = [cqi_est, snr_linear, curr_context_id, min_acc.item()]
        if total_delay > max_delay:
            self.delay_vio_num += 1

        done = self.step_num >= self.slot_num - 1
        if done:
            self.episode_num += 1
            episode_total_delay = np.sum(self.total_delay_list) / self.slot_num
            episode_total_energy = np.sum(self.total_energy_list) / self.slot_num
            episode_acc_exp = np.sum(self.acc_exp_list) / self.slot_num
            episode_reward = np.sum(self.reward_list) / self.slot_num
            episode_re_trans_num = np.sum(self.re_trans_list) / self.slot_num
            episode_acc_vio = np.sum(self.acc_vio_list) / max(self.acc_vio_num, 1)
            self.acc_vio_num /= self.slot_num

            print("Episode index:", self.episode_num)
            print("Average total delay (s):", episode_total_delay)
            print("Average energy (J):", episode_total_energy, "Remaining energy:", self.remain_energy)
            print("Average accuracy:", episode_acc_exp)
            print("Average reward:", episode_reward)
            print("Delay violations:", self.delay_vio_num)
            print("Retransmissions:", episode_re_trans_num)
            print("Accuracy violation rate:", self.acc_vio_num)
            print("Accuracy violation:", episode_acc_vio)

            self.episode_total_delay_list.append(episode_total_delay)
            self.episode_total_energy_list.append(episode_total_energy)
            self.episode_acc_exp_list.append(episode_acc_exp)
            self.episode_reward_list.append(episode_reward)
            self.episode_delay_vio_num_list.append(self.delay_vio_num)
            self.episode_remain_energy_list.append(self.remain_energy)
            self.episode_re_trans_num_list.append(episode_re_trans_num)
            self.episode_acc_vio_num_list.append(self.acc_vio_num)
            self.episode_acc_vio_list.append(episode_acc_vio)

        self.step_num += 1
        return np.array(state), reward, done
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
        self.acc_vio_list = np.zeros([1, self.slot_num])
        self.action_freq_list = np.zeros([1, 31])
        self.bad_action_freq_list = np.zeros([1, 31])
        self.episode_total_delay_list = []
        self.episode_total_energy_list = []
        self.episode_acc_exp_list = []
        self.episode_reward_list = []
        self.episode_delay_vio_num_list = []
        self.episode_remain_energy_list = []
        self.episode_re_trans_num_list = []
        self.episode_acc_vio_num_list = []
        self.episode_acc_vio_list = []
        return np.array(state_init)

    def input_est_err(self, est_err_para):
        self.est_err_para = est_err_para

        import pandas as pd  # Ensure this is at the top of your file

        self.mcs_df = pd.read_csv("/home/ababu/mcs_performance_table.csv")

        self.mcs_models = {}
        self.mcs_efficiencies = {}

        for mcs_index, group in self.mcs_df.groupby("MCS_Index"):
            sinr = group["SINR_dB"].values
            ber = group["BER"].values
            eff = group["Spectral_Efficiency_bpsHz"].values[0]
            coeffs = np.polyfit(sinr, ber, deg=5)
            self.mcs_models[mcs_index] = np.poly1d(coeffs)
            self.mcs_efficiencies[mcs_index] = eff

    def input_snr(self, snr_db):
        self.target_snr_db = snr_db

    def get_action_name(self, action):
        (self.action_sunny_list, self.action_rain_list, self.action_snow_list,
         self.action_motorway_list, self.action_fog_list, self.action_night_list) = util.action_gen()
        action_info = util.action_mapping(self.action_sunny_list, self.action_rain_list, self.action_snow_list,
                                          self.action_motorway_list, self.action_fog_list, self.action_night_list,
                                          self.curr_context, action)

        action_name = str(action_info.fusion_name) + "+Res" + action_info.backbone
        return action_name

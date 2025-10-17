import math
import gym
from gym import spaces, logger
import numpy as np
import os
import re
from scipy.io import loadmat
from tensorboard.compat.tensorflow_stub.dtypes import float32
import HERACLES.envs.utils_proposed as util
import random
import matplotlib.pyplot as plt
import scipy.special as ss
import pandas as pd
from scipy.interpolate import interp1d
import pandas as pd  # Add this at the top if not already present

class EnvProposed_origin(gym.Env):
    def __init__(self):
        self.name = "proposed_origin"
        self.slot_num = 3000
        self.max_energy = 2000
        self.remain_energy = self.max_energy
        self.last_energy = 0
        self.sensor_num = 4
        self.bandwidth = 20e6
        self.max_power = 1
        self.est_err_para = 0.5
        self.data_size = np.zeros([1, self.sensor_num])
        for i in range(self.sensor_num):
            self.data_size[0, i] = np.random.rand(1) * 10000
        self.sub_block_length = 128
        self.step_reward_list = []
        self.target_snr_db = 2
        self.total_delay_list = np.zeros([1, self.slot_num])
        self.total_energy_list = np.zeros([1, self.slot_num])
        self.acc_exp_list = np.zeros([1, self.slot_num])
        self.reward_list = np.zeros([1, self.slot_num])
        self.re_trans_list = np.zeros([1, self.slot_num])
        self.acc_vio_list = np.zeros([1, self.slot_num])
        self.episode_total_delay_list = []
        self.episode_total_energy_list = []
        self.episode_acc_exp_list = []
        self.episode_reward_list = []
        self.episode_delay_vio_num_list = []
        self.episode_remain_energy_list = []
        self.episode_re_trans_num_list = []
        self.episode_acc_vio_num_list = []
        self.episode_acc_vio_list = []
        np.seterr(over='ignore')
        self.context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
        self.context_prob = [0.05, 0.05, 0.2, 0.1, 0.2, 0.4]
        self.context_interval = 100
        self.context_num = int(self.slot_num / self.context_interval)
        self.context_train_list = np.random.choice(list(range(len(self.context_list))), size=self.context_num,
                                                   p=self.context_prob)
        self.delay_vio_num = 0
        self.context_flag = 0
        self.num_actions = 31  # Assuming MCS indices 0–30
        self.show_fit_plot = False
        self.curr_context = None
        self.enable_re_trans = True
        self.re_trans_num = -1
        self.done = False
        (self.action_sunny_list, self.action_rain_list, self.action_snow_list,
         self.action_motorway_list, self.action_fog_list, self.action_night_list) = util.action_gen()
        self.action_space = spaces.Discrete(self.num_actions)
        self.action_freq_list = np.zeros([1, self.num_actions])
        self.bad_action_freq_list = np.zeros([1, self.num_actions])
        obs_low = np.array([1, 0, 0, 0])
        obs_high = np.array([15, 20, 5, 1])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.step_num = 0
        self.episode_num = 0
        self.max_re_trans_num = 500
        self.kappa_1 = 2
        self.kappa_2 = 1
        self.kappa_3 = 1
        self.acc_vio_num = 0
        self.action_name = "None"

        # Load MCS performance table from CSV
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

        mcs_index = action  # Direct mapping
        snr_db = self.target_snr_db
        snr_db = np.clip(snr_db, self.mcs_df["SINR_dB"].min(), self.mcs_df["SINR_dB"].max())
        snr_linear = 10 ** (snr_db / 10)

    # BER and spectral efficiency from CSV-based models
        ber_model = self.mcs_models.get(mcs_index, None)
        ber = np.clip(ber_model(snr_db), 0.00001, 0.99999) if ber_model else 0.5
        spectral_eff = self.mcs_efficiencies.get(mcs_index, 0.5)
        trans_rate = spectral_eff * self.bandwidth  # bits/sec

        re_trans_delay = 0
        re_trans_energy = 0

        if len(action_info.fusion_name) == 1:
            data_size_idx = action_info.fusion_name[0] - 1
            data_size = self.data_size[0, data_size_idx]
            block_num = np.floor(data_size / self.sub_block_length)

            if self.enable_re_trans:
                self.re_trans_num = 0
                per = 1 - (1 - ber) ** self.sub_block_length
                for _ in range(int(block_num)):
                    re_trans_num_block = 0
                    is_trans_success = 0
                    while is_trans_success == 0:
                        is_trans_success = random.choices([0, 1], weights=[per, 1 - per])[0]
                        if is_trans_success == 1 or re_trans_num_block >= self.max_re_trans_num:
                            break
                        else:
                            re_trans_num_block += 1
                            per = 1 - (1 - ber) ** (self.sub_block_length / (1 - spectral_eff / np.log2(1 + snr_linear)))
                    self.re_trans_num += re_trans_num_block
                re_trans_delay = self.re_trans_num * ((1 / spectral_eff - 1) * self.sub_block_length / trans_rate)
                re_trans_energy = self.max_power * re_trans_delay

            trans_delay = data_size / trans_rate + re_trans_delay

        else:
            data_size_idx = [i - 1 for i in action_info.fusion_name]
            data_size = np.max(self.data_size[0, data_size_idx])
            block_num = np.floor(data_size / self.sub_block_length)

            if self.enable_re_trans:
                self.re_trans_num = 0
                per = 1 - (1 - ber) ** self.sub_block_length
                for _ in range(int(block_num)):
                    re_trans_num_block = 0
                    is_trans_success = 0
                    while is_trans_success == 0:
                        is_trans_success = random.choices([0, 1], weights=[per, 1 - per])[0]
                        if is_trans_success == 1 or re_trans_num_block >= self.max_re_trans_num:
                            break
                        else:
                            re_trans_num_block += 1
                            per = 1 - (1 - ber) ** (self.sub_block_length / (1 - spectral_eff / np.log2(1 + snr_linear)))
                    self.re_trans_num += re_trans_num_block
                re_trans_delay = self.re_trans_num * ((1 / spectral_eff - 1) * self.sub_block_length / trans_rate)
                re_trans_energy = self.max_power * re_trans_delay

            trans_delay = data_size / trans_rate + re_trans_delay

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
        if acc_exp < min_acc:
            self.acc_vio_num += 1
            self.bad_action_freq_list[0, action] += 1
            self.acc_vio_list[0, self.step_num] = np.abs(acc_exp - min_acc)

        self.acc_exp_list[0, self.step_num] = acc_exp
        reward_1 = acc_exp - min_acc
        reward_2 = total_delay / max_delay
        reward_3 = total_energy / self.max_energy
        reward = self.kappa_1 * reward_1 - self.kappa_2 * reward_2 - self.kappa_3 * reward_3
        self.reward_list[0, self.step_num] = reward
        self.step_reward_list.append(reward.item())

        state = [cqi_est, snr_linear, curr_context_id, min_acc.item()]
        if total_delay > max_delay:
            self.delay_vio_num += 1

        if self.step_num >= self.slot_num - 1:
            self.done = True
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
            self.episode_remain_energy_list.append(self.remain_energy.item())
            self.episode_re_trans_num_list.append(episode_re_trans_num)
            self.episode_acc_vio_num_list.append(self.acc_vio_num)
            self.episode_acc_vio_list.append(episode_acc_vio)

        self.step_num += 1
        self.action_freq_list[0, action] += 1

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
        self.remain_energy = self.max_energy
        self.done = False
        self.total_delay_list = np.zeros([1, self.slot_num])
        self.total_energy_list = np.zeros([1, self.slot_num])
        self.acc_exp_list = np.zeros([1, self.slot_num])
        self.reward_list = np.zeros([1, self.slot_num])
        self.re_trans_list = np.zeros([1, self.slot_num])
        self.acc_vio_list = np.zeros([1, self.slot_num])
        self.action_freq_list = np.zeros([1, self.num_actions])
        self.bad_action_freq_list = np.zeros([1, self.num_actions])
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

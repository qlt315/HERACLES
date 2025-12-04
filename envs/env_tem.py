import math
import gym
from gym import spaces, logger
import numpy as np
import os
import re
from scipy.io import loadmat
from tensorboard.compat.tensorflow_stub.dtypes import float32
import HERACLES.envs.utils_tem as util
import random
import matplotlib.pyplot as plt
import scipy.special as ss
import pandas as pd
# import envs.mobile_channel_gen

class EnvTEM(gym.Env):
    def __init__(self):
        self.name = "tem"
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
        self.acc_vio_list = np.zeros([1, self.slot_num])
        self.min_acc_list = np.zeros([1, self.slot_num])

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
        self.max_re_trans_num = 500
        self.kappa_1 = 2  # acc reward coefficient
        self.kappa_2 = 1  # delay reward coefficient
        self.kappa_3 = 1  # energy consumption reward coefficient
        self.acc_vio_num = 0
        self.action_name = "None"
        # Data loading and fitting
        # Load HM data
        self.mcs_df = pd.read_csv("/home/ababu/mcs_performance_table.csv")
        #if not os.path.isfile(mcs_csv_path):
            # Try current directory fallback
         #   mcs_csv_path = "/home/ababu/mcs_performance_table.csv"
        #if not os.path.isfile(mcs_csv_path):
         #   raise FileNotFoundError(f"MCS CSV not found at expected locations. Please place at {mcs_csv_path}")

        self.mcs_df = pd.read_csv("/home/ababu/mcs_performance_table.csv", sep=None, engine='python')  # let pandas infer delimiter
        # Clean column names in case of whitespace / BOM
        self.mcs_df.columns = [c.strip() for c in self.mcs_df.columns]

        # ensure proper dtypes
        # expected columns: 'SINR_dB', 'MCS_Index', 'BER', 'Spectral_Efficiency_bpsHz'
        for col in ['SINR_dB', 'MCS_Index', 'BER', 'Spectral_Efficiency_bpsHz']:
            if col not in self.mcs_df.columns:
                raise ValueError(f"Expected column '{col}' in MCS CSV.")
        self.mcs_df['SINR_dB'] = self.mcs_df['SINR_dB'].astype(float)
        self.mcs_df['MCS_Index'] = self.mcs_df['MCS_Index'].astype(int)
        self.mcs_df['BER'] = self.mcs_df['BER'].astype(float)
        self.mcs_df['Spectral_Efficiency_bpsHz'] = self.mcs_df['Spectral_Efficiency_bpsHz'].astype(float)

        # Precompute available MCS indices from CSV
        self.available_mcs = np.unique(self.mcs_df['MCS_Index'].values)


        wireless_data_path = '/home/ababu/HERACLES/system_data/5G_dataset/Netflix/Driving/animated-RickandMorty'
        self.snr_array, self.cqi_array = util.obtain_cqi_and_snr(wireless_data_path, self.slot_num)

    
    def step(self, action):
        max_delay = np.random.uniform(low=0.3, high=1, size=1)
        curr_context_id = self.context_train_list[self.context_flag]
        self.curr_context = self.context_list[curr_context_id]
        if self.step_num % self.context_interval == 0 and self.step_num != 0:
            self.context_flag = self.context_flag + 1
            curr_context_id = self.context_train_list[self.context_flag]
            self.curr_context = self.context_list[curr_context_id]

        min_acc = util.obtain_min_acc(self.curr_context)
        self.min_acc_list[0, self.step_num] = min_acc
        cqi = int(self.cqi_array[self.step_num])
        cqi_est = util.estimate_cqi(cqi, self.est_err_para)

        action_info = util.action_mapping(self.action_sunny_list, self.action_rain_list, self.action_snow_list,
                                          self.action_motorway_list, self.action_fog_list, self.action_night_list,
                                          self.curr_context, action)
        requested_mcs = int(action)  # action is directly treated as MCS index
        if requested_mcs not in self.available_mcs:
            nearest_idx = np.argmin(np.abs(self.available_mcs - requested_mcs))
            chosen_mcs = int(self.available_mcs[nearest_idx])
        else:
            chosen_mcs = requested_mcs

        snr_db = float(self.target_snr_db)
        snr_linear = 10 ** (snr_db / 10)
        # Filter table for the chosen MCS
        df_mcs = self.mcs_df[self.mcs_df['MCS_Index'] == chosen_mcs]

        if df_mcs.empty:
            # Fail-safe: pick row with overall nearest MCS+SINR in table
            all_diff = np.abs(self.mcs_df['MCS_Index'] - requested_mcs) + np.abs(self.mcs_df['SINR_dB'] - snr_db) * 0.01
            row = self.mcs_df.iloc[int(np.argmin(all_diff.values))]
            mcs_ber = float(row['BER'])
            spectral_eff = float(row['Spectral_Efficiency_bpsHz'])
        else:
            # select the row in df_mcs where SINR_dB is closest to snr_db
            idx = int(np.argmin(np.abs(df_mcs['SINR_dB'].values - snr_db)))
            row = df_mcs.iloc[idx]
            mcs_ber = float(row['BER'])
            spectral_eff = float(row['Spectral_Efficiency_bpsHz'])

        # Transmission rate (bits/s) from spectral efficiency (bits/s/Hz)
        tm_trans_rate = spectral_eff * self.bandwidth  # bits / s
        tm_ber = np.clip(mcs_ber, 1e-12, 0.999999)  # keep in bounds

        # Calculate PER and retransmission similar to before
        re_trans_energy = 0
        re_trans_delay = 0
        data_size_idx = action_info.fusion_name[0] - 1
        data_size = self.data_size[0, data_size_idx]  # bytes? earlier code used / self.tm_coding_rate
        # Assuming data_size in bits required; if bytes then multiply by 8
        # To be consistent with previous code where they divided by coding rate, we simply treat data_size as bits here.
        # If your data_size is bytes, use data_size *= 8

        # Use block size self.sub_block_length (bits) as before
        block_num = 1
        if self.enable_re_trans:
            self.re_trans_num = 0
            # number of blocks (floor)
            block_num = max(1, int(np.floor(data_size / self.sub_block_length)))
            tm_per = 1 - (1 - tm_ber) ** self.sub_block_length
            for j in range(int(block_num)):
                re_trans_num_block = 0
                is_trans_success = 0
                while is_trans_success == 0:
                    is_trans_success = random.choices([0, 1], weights=[tm_per, 1 - tm_per])[0]
                    if is_trans_success == 1 or re_trans_num_block >= self.max_re_trans_num:
                        break
                    else:
                        re_trans_num_block += 1
                        tm_per = 1 - (1 - tm_ber) ** max(1, int(self.sub_block_length / 1))  # kept simple
                self.re_trans_num += re_trans_num_block
            # compute retrans delay & energy using tm_trans_rate
            re_trans_delay = self.re_trans_num * ((1 / 1 - 1) * self.sub_block_length / tm_trans_rate) if tm_trans_rate > 0 else 0
            # note: (1/1 -1) = 0, so re_trans_delay simplified; keep base cost as retransmissions cost proportional to block length/time
            # Use simpler estimate: each retrans block costs block transmission time
            if tm_trans_rate > 0:
                re_trans_delay = self.re_trans_num * (self.sub_block_length / tm_trans_rate)
            re_trans_energy = self.max_power * re_trans_delay

        # Transmission delay
        trans_delay = 0
        if tm_trans_rate > 0:
            trans_delay = data_size / tm_trans_rate
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

        reward_1 = ss.erf(acc_exp - min_acc)
        reward_2 = total_delay / max_delay
        reward_3 = total_energy / self.max_energy
        reward = self.kappa_1 * reward_1 - self.kappa_2 * reward_2 - self.kappa_3 * reward_3
        self.reward_list[0, self.step_num] = reward
        self.step_reward_list.append(reward.item())

        state = [cqi_est, snr_linear, curr_context_id, min_acc.item()]
        if total_delay > max_delay:
            self.delay_vio_num += 1

        if self.step_num >= self.slot_num - 1:
            self.episode_num += 1
            self.done = True
            episode_total_delay = np.sum(self.total_delay_list) / self.slot_num
            episode_total_energy = np.sum(self.total_energy_list) / self.slot_num
            episode_acc_exp = np.sum(self.acc_exp_list) / self.slot_num
            episode_reward = np.sum(self.reward_list) / self.slot_num
            episode_re_trans_num = np.sum(self.re_trans_list) / self.slot_num
            episode_acc_vio = np.sum(self.acc_vio_list) / max(self.acc_vio_num, 1)
            self.acc_vio_num /= self.slot_num

            print("Episode index:", self.episode_num)
            print("Average total delay (s):", episode_total_delay)
            print("Average total energy consumption (J)", episode_total_energy, "Remain energy (J)", self.remain_energy)
            print("Average accuracy expectation", episode_acc_exp)
            print("Average episode reward", episode_reward)
            print("Delay violation slot number:", self.delay_vio_num)
            print("Retransmission number:", episode_re_trans_num)
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
        self.action_freq_list = np.zeros([1, self.num_actions])  # Use self.num_actions = 31 if MCS 0–30
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

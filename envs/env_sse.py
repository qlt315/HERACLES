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
import envs.utils_sse as util
import matplotlib.pyplot as plt


class EnvSSE(gym.Env):
    def __init__(self):
        self.name = "sse"
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
        self.data_size = np.zeros([1, self.sensor_num])
        for i in range(self.sensor_num):
            self.data_size[0, i] = np.random.rand(1) * 5000
        # self.data_size[0, 0] = 6064128
        # self.data_size[0, 1] = self.data_size[0, 0]
        # self.data_size[0, 2] = 1.8432e6
        # self.data_size[0, 3] = 1.73e6

        self.target_snr_db = 10
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
        self.context_list = ["snow", "fog", "motorway", "night", "rain", "sunny"]
        # self.context_list = ["sunny", "sunny", "sunny", "sunny", "sunny", "sunny"]
        self.context_prob = [0.05, 0.05, 0.2, 0.1, 0.2, 0.4]
        self.context_interval = 100  # Interval for context to change
        self.context_num = int(self.slot_num / self.context_interval)
        self.context_train_list = np.random.choice(list(range(len(self.context_list))),size=self.context_num, p=self.context_prob)
        self.context_flag = 0
        self.delay_vio_num = 0

        self.curr_context = None
        self.show_fit_plot = False
        self.re_trans_num = -1
        self.enable_re_trans = True  # Consider retransmission or not
        self.sub_block_length = 128
        # DRL parameter settings
        (self.action_sunny_list, self.action_rain_list, self.action_snow_list,
         self.action_motorway_list, self.action_fog_list, self.action_night_list) = util.action_gen()
        self.action_space = spaces.Discrete(24)

        # Obs: (1) Estimated channel (2) Task context (0-5) (3) Remaining energy (4) Maximum tolerant delay
        obs_low = np.array([0, 0, 0, 0])
        obs_high = np.array([5, 5, 10000, 10])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.step_num = 0
        self.episode_num = 0

        self.kappa_1 = 1  # delay reward coefficient
        self.kappa_2 = 1  # energy consumption reward coefficient

        # Data loading and fitting
        # Load HM data
        self.hm_folder_path = "./system_data/hierarchical modulation/four_layers_data"
        hm_pattern = r'^snr_([\d.]+)_([\d.]+)_([\d.]+)_bpsk(_bpsk){3}_esterr_([\d\.]+)_rate_([\d\.]+)_power_ratio_([\d\.]+)_([\d\.]+)_([\d\.]+)_([\d\.]+)\.mat'
        for filename in os.listdir(self.hm_folder_path):
            if filename.endswith('.mat'):
                match = re.match(hm_pattern, filename)
                if match:
                    self.hm_err_para_ = float(match.group(5))
                    if self.hm_err_para_ != self.est_err_para:
                        continue
                    else:
                        print("Loading 4-Layer HM SNR-BER data (QPSK-QPSK-QPSK-QPSK):", filename)
                        hm_file_path = os.path.join(self.hm_folder_path, filename)
                        self.hm_snr_min = float(match.group(1))
                        self.hm_snr_int = float(match.group(2))
                        self.hm_snr_max = float(match.group(3))
                        self.hm_coding_rate = float(match.group(6))
                        self.hm_power_ratio = [float(match.group(7)), float(match.group(8)), float(match.group(9)),
                                               float(match.group(10))]

                        self.hm_data = loadmat(hm_file_path, variable_names=['ber'])
                        self.hm_data['ber'] = self.hm_data['ber'].T
                        self.hm_ber_1 = self.hm_data['ber'][0, :]  # Layer 1 BER

                        self.hm_ber_2 = self.hm_data['ber'][1, :]  # Layer 2 BER with SIC
                        self.hm_ber_3 = self.hm_data['ber'][2, :]  # Layer 3 BER with SIC
                        self.hm_ber_4 = self.hm_data['ber'][3, :]  # Layer 4 BER with SIC

                        self.hm_ber_1 = self.hm_ber_1[self.hm_ber_1 != 0]
                        self.hm_ber_2 = self.hm_ber_2[self.hm_ber_2 != 0]
                        self.hm_ber_3 = self.hm_ber_3[self.hm_ber_3 != 0]
                        self.hm_ber_4 = self.hm_ber_4[self.hm_ber_4 != 0]

                        self.hm_snr_list = np.arange(self.hm_snr_min, self.hm_snr_max + self.hm_snr_int,
                                                     self.hm_snr_int)
                        self.hm_snr_list = self.hm_snr_list[:len(self.hm_ber_1)]
                        # Function fitting
                        self.hm_degree = 5

                        self.hm_coefficients_1 = np.polyfit(self.hm_snr_list.ravel(), self.hm_ber_1.ravel(),
                                                            self.hm_degree)
                        self.hm_coefficients_2 = np.polyfit(self.hm_snr_list.ravel(), self.hm_ber_2.ravel(),
                                                            self.hm_degree)
                        self.hm_coefficients_3 = np.polyfit(self.hm_snr_list.ravel(), self.hm_ber_3.ravel(),
                                                            self.hm_degree)
                        self.hm_coefficients_4 = np.polyfit(self.hm_snr_list.ravel(), self.hm_ber_4.ravel(),
                                                            self.hm_degree)
                        self.hm_polynomial_model_1 = np.poly1d(self.hm_coefficients_1)
                        self.hm_polynomial_model_2 = np.poly1d(self.hm_coefficients_2)
                        self.hm_polynomial_model_3 = np.poly1d(self.hm_coefficients_3)
                        self.hm_polynomial_model_4 = np.poly1d(self.hm_coefficients_4)

                        if self.show_fit_plot:
                            # Generate fitting curves
                            snr_fit = np.linspace(np.min(self.hm_snr_list.ravel()), np.max(self.hm_snr_list.ravel()),
                                                  100)
                            ber_fit_1 = self.hm_polynomial_model_1(snr_fit)
                            ber_fit_2 = self.hm_polynomial_model_2(snr_fit)
                            ber_fit_3 = self.hm_polynomial_model_3(snr_fit)
                            ber_fit_4 = self.hm_polynomial_model_4(snr_fit)

                            # Validate the fitting results and plot
                            plt.scatter(self.hm_snr_list.ravel(), self.hm_ber_1, color='red',
                                        label='original layer 1 BER')
                            plt.scatter(self.hm_snr_list.ravel(), self.hm_ber_2, color='green',
                                        label='original layer 2 BER')
                            plt.scatter(self.hm_snr_list.ravel(), self.hm_ber_3, color='black',
                                        label='original layer 3 BER')
                            plt.scatter(self.hm_snr_list.ravel(), self.hm_ber_4, color='orange',
                                        label='original layer 4 BER')
                            plt.plot(snr_fit, ber_fit_1, label='fitting curve for layer 1 BER', color='blue')
                            plt.plot(snr_fit, ber_fit_2, label='fitting curve for layer 2 BER', color='blue')
                            plt.plot(snr_fit, ber_fit_3, label='fitting curve for layer 3 BER', color='blue')
                            plt.plot(snr_fit, ber_fit_4, label='fitting curve for layer 4 BER', color='blue')
                            plt.xlabel('SNR (dB)')
                            plt.ylabel('BER')
                            plt.title('SNR vs BER fitting (HM)')
                            plt.legend()
                            plt.grid(True)
                            plt.show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # print("Episode index:", self.episode_num, "Step index:", self.step_num)
        max_delay = 0.5
        curr_context_id = self.context_train_list[self.context_flag]
        self.curr_context = self.context_list[curr_context_id]
        if self.step_num % self.context_interval == 0 and self.step_num != 0:
            self.context_flag = self.context_flag + 1
            curr_context_id = self.context_train_list[self.context_flag]
            self.curr_context = self.context_list[curr_context_id]
        h = (np.random.randn(1) + 1j  # Wireless channel
             * np.random.randn(1)) / np.sqrt(2)

        # Channel estimation
        est_err = self.est_err_para * np.abs(h)
        h_est = h + est_err + est_err * 1j
        noise_power = abs(h_est) * self.max_power / (10 ** (self.target_snr_db / 10))

        action_info = util.action_mapping(self.action_sunny_list, self.action_rain_list, self.action_snow_list,
                                          self.action_motorway_list, self.action_fog_list, self.action_night_list,
                                          self.curr_context, action)
        # Calculate SNR and trans rate
        hm_snr = abs(h_est) * self.max_power / noise_power
        hm_snr_db = 10 * np.log10(hm_snr)
        hm_snr_db = np.clip(hm_snr_db, np.min(self.hm_snr_list), np.max(self.hm_snr_list))

        # power ratio = 0.5 / 0.4 / 0.3 / 0.2
        hm_snr_1 = abs(h_est) * (0.5 * self.max_power) / noise_power
        hm_snr_2 = abs(h_est) * (0.4 * self.max_power) / noise_power
        hm_snr_3 = abs(h_est) * (0.3 * self.max_power) / noise_power
        hm_snr_4 = abs(h_est) * (0.2 * self.max_power) / noise_power

        hm_snr_1_db = 10 * np.log10(hm_snr_1)
        hm_snr_2_db = 10 * np.log10(hm_snr_2)
        hm_snr_3_db = 10 * np.log10(hm_snr_3)
        hm_snr_4_db = 10 * np.log10(hm_snr_4)

        hm_snr_1_db = np.clip(hm_snr_1_db, np.min(self.hm_snr_list), np.max(self.hm_snr_list))
        hm_snr_2_db = np.clip(hm_snr_2_db, np.min(self.hm_snr_list), np.max(self.hm_snr_list))
        hm_snr_3_db = np.clip(hm_snr_3_db, np.min(self.hm_snr_list), np.max(self.hm_snr_list))
        hm_snr_4_db = np.clip(hm_snr_4_db, np.min(self.hm_snr_list), np.max(self.hm_snr_list))

        hm_ber_1 = np.clip(self.hm_polynomial_model_1(hm_snr_1_db), 0.00001, 0.99999)
        hm_ber_2 = np.clip(self.hm_polynomial_model_2(hm_snr_2_db), 0.00001, 0.99999)
        hm_ber_3 = np.clip(self.hm_polynomial_model_3(hm_snr_3_db), 0.00001, 0.99999)
        hm_ber_4 = np.clip(self.hm_polynomial_model_4(hm_snr_4_db), 0.00001, 0.99999)

        hm_trans_rate_1 = self.bandwidth * np.log2(1 + hm_snr_1)  # Bit / s
        hm_trans_rate_2 = self.bandwidth * np.log2(1 + hm_snr_2)  # Bit / s
        hm_trans_rate_3 = self.bandwidth * np.log2(1 + hm_snr_3)  # Bit / s
        hm_trans_rate_4 = self.bandwidth * np.log2(1 + hm_snr_4)  # Bit / s

        # Calculate accuracy expectation
        # acc_exp = acc_exp_gen(per, self.curr_context)
        # acc_exp = action_info.acc * (1-(per_list[action].item()))
        acc_exp = action_info.acc
        self.acc_exp_list[0, self.step_num] = acc_exp

        # Calculate delay
        order = action_info.fusion_name
        # hm_data_size = np.floor(np.max(self.data_size) / self.hm_coding_rate)
        # trans_delay = hm_data_size / hm_trans_rate
        trans_delay_1 = np.floor(self.data_size[0, order[0] - 1] / self.hm_coding_rate) / hm_trans_rate_1
        trans_delay_2 = np.floor(self.data_size[0, order[0] - 1] / self.hm_coding_rate) / hm_trans_rate_2
        trans_delay_3 = np.floor(self.data_size[0, order[0] - 1] / self.hm_coding_rate) / hm_trans_rate_3
        trans_delay_4 = np.floor(self.data_size[0, order[0] - 1] / self.hm_coding_rate) / hm_trans_rate_4

        # Retransmission simulation
        if self.enable_re_trans:
            re_trans_num_1 = 0
            re_trans_num_2 = 0
            re_trans_num_3 = 0
            re_trans_num_4 = 0

            block_num_1 = np.floor(self.data_size[0, order[0] - 1] / self.hm_coding_rate / self.sub_block_length)
            block_num_2 = np.floor(self.data_size[0, order[1] - 1] / self.hm_coding_rate / self.sub_block_length)
            block_num_3 = np.floor(self.data_size[0, order[2] - 1] / self.hm_coding_rate / self.sub_block_length)
            block_num_4 = np.floor(self.data_size[0, order[3] - 1] / self.hm_coding_rate / self.sub_block_length)

            per_1 = 1 - ((1 - hm_ber_1) ** self.sub_block_length)
            per_2 = 1 - ((1 - hm_ber_2) ** self.sub_block_length)
            per_3 = 1 - ((1 - hm_ber_3) ** self.sub_block_length)
            per_4 = 1 - ((1 - hm_ber_4) ** self.sub_block_length)

            # Calculate PER
            # per_list = per_list_gen(hm_ber_list, self.data_size, self.hm_coding_rate)
            # per_curr = 1 - ((1 - hm_ber_1) ** self.sub_block_length * (1 - hm_ber_2) ** self.sub_block_length
            #                 * (1 - hm_ber_3) ** self.sub_block_length * (1 - hm_ber_4) ** self.sub_block_length)
            # print("Using HM")
            # print("BER 1:", hm_ber_1, "BER 2", hm_ber_2)
            # for b_1 in range(int(block_num_1)):
            #     is_trans_success = 0
            #     while is_trans_success == 0:
            #         # Generate 1 (success) with probability 1-p and 0 (fail) with p
            #         is_trans_success = \
            #             random.choices([0, 1], weights=[per_1, 1 - per_1])[0]
            #         if is_trans_success == 1:
            #             continue
            #         else:
            #             re_trans_num_1 = re_trans_num_1 + 1
            #             if re_trans_num_1 >= 500 * hm_ber_1:
            #                 break
            #
            # for b_2 in range(int(block_num_2)):
            #     is_trans_success = 0
            #     while is_trans_success == 0:
            #         # Generate 1 (success) with probability 1-p and 0 (fail) with p
            #         is_trans_success = \
            #             random.choices([0, 1], weights=[per_2, 1 - per_2])[0]
            #         if is_trans_success == 1:
            #             continue
            #         else:
            #             re_trans_num_2 = re_trans_num_2 + 1
            #             if re_trans_num_2 >= 50000 * hm_ber_2:
            #                 break
            #
            # for b_3 in range(int(block_num_3)):
            #     is_trans_success = 0
            #     while is_trans_success == 0:
            #         # Generate 1 (success) with probability 1-p and 0 (fail) with p
            #         is_trans_success = \
            #             random.choices([0, 1], weights=[per_3, 1 - per_3])[0]
            #         if is_trans_success == 1:
            #             continue
            #         else:
            #             re_trans_num_3 = re_trans_num_3 + 1
            #             if re_trans_num_3 >= 50000 * hm_ber_3:
            #                 break
            #
            # for b_4 in range(int(block_num_4)):
            #     is_trans_success = 0
            #     while is_trans_success == 0:
            #         # Generate 1 (success) with probability 1-p and 0 (fail) with p
            #         is_trans_success = \
            #             random.choices([0, 1], weights=[per_4, 1 - per_4])[0]
            #         if is_trans_success == 1:
            #             continue
            #         else:
            #             re_trans_num_4 = re_trans_num_4 + 1
            #             if re_trans_num_4 >= 50000 * hm_ber_4:
            #                 break
            re_trans_num_1 = 5e2 * hm_ber_1
            re_trans_num_2 = 5e2 * hm_ber_2
            re_trans_num_3 = 5e2 * hm_ber_3
            re_trans_num_4 = 5e2 * hm_ber_4
            self.re_trans_num = re_trans_num_1 + re_trans_num_2 + re_trans_num_3 + re_trans_num_4
            re_trans_delay_1 = re_trans_num_1 * (self.sub_block_length / hm_trans_rate_1)
            re_trans_delay_2 = re_trans_num_2 * (self.sub_block_length / hm_trans_rate_2)
            re_trans_delay_3 = re_trans_num_3 * (self.sub_block_length / hm_trans_rate_3)
            re_trans_delay_4 = re_trans_num_4 * (self.sub_block_length / hm_trans_rate_4)
            trans_delay_1 = trans_delay_1 + re_trans_delay_1
            trans_delay_2 = trans_delay_2 + re_trans_delay_2
            trans_delay_3 = trans_delay_3 + re_trans_delay_3
            trans_delay_4 = trans_delay_4 + re_trans_delay_4
        # print("Retrains delay:",re_trans_delay_1,re_trans_delay_2,re_trans_delay_3,re_trans_delay_4)
        trans_delay = max(trans_delay_1, trans_delay_2, trans_delay_3, trans_delay_4)
        # print(trans_delay_1,trans_delay_2,trans_delay_3,trans_delay_4)
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
        # # Option 1: Calculate accuracy (expectation)
        # acc_exp = util.acc_exp_gen(per_list, self.curr_context)
        # self.acc_exp_list[0, self.step_num] = acc_exp
        # # Normalize the reward (Acc reward should be further normalized)
        # acc_exp = util.acc_normalize(acc_exp, self.curr_context)

        # Option 2: Don't consider expectation
        acc_exp = action_info.acc
        acc_exp = util.acc_normalize(acc_exp, self.curr_context)

        reward_1 = acc_exp
        reward_2 = (max_delay - total_delay) / max_delay
        # reward_3 = self.remain_energy / self.max_energy
        reward_3 = total_energy
        reward = reward_1 + self.kappa_1 * reward_2 - self.kappa_2 * reward_3

        # print("slot index:", self.step_num, "action info:", action_info.fusion_name, "branch:", action_info.backbone,
        #       "reward:", reward, "reward 1:", reward_1, "reward_2:", reward_2, "reward_3", -reward_3, "trans delay:", trans_delay,
        #       "context:",self.curr_context)

        # reward = acc_exp + self.kappa_1 * (max_delay - total_delay) + self.kappa_2 * self.remain_energy
        self.reward_list[0, self.step_num] = reward
        self.step_reward_list.append(reward.item())
        # State calculation
        state = [np.abs(h_est).item(), curr_context_id, self.remain_energy.item(), max_delay]
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
        h_init = 0.5 + 0.5j
        context_id_init = 1
        max_delay_init = 1
        state_init = [np.abs(h_init), context_id_init, self.max_energy, max_delay_init]
        self.context_flag = 0
        self.step_num = 0
        self.delay_vio_num = 0
        self.remain_energy = self.max_energy  # Available energy of current slot
        return np.array(state_init)

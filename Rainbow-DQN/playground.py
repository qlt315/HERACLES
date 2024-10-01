import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import os
import re
from scipy.io import loadmat
import envs.utils_proposed as util
import random
import matplotlib.pyplot as plt


class EnvProposed(gym.Env):
    def __init__(self):
        self.name = "proposed"
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
        self.step_reward_list = []
        # self.data_size = np.zeros([1, 4])
        # self.data_size[0, 0] = 6064128
        # self.data_size[0, 1] = self.data_size[0, 0]
        # self.data_size[0, 2] = 1.8432e6
        # self.data_size[0, 3] = 1.73e6

        self.total_delay_list = np.zeros([1, self.slot_num])
        self.total_energy_list = np.zeros([1, self.slot_num])
        self.acc_exp_list = np.zeros([1, self.slot_num])
        self.reward_list = np.zeros([1, self.slot_num])

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

        self.show_fit_plot = True
        self.curr_context = None
        self.enable_re_trans = False  # Consider retransmission or not
        self.re_trans_num = -1

        # DRL parameter settings
        (self.action_sunny_list, self.action_rain_list, self.action_snow_list,
         self.action_motorway_list, self.action_fog_list, self.action_night_list) = util.action_gen()
        self.action_space = spaces.Discrete(33)

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
        self.hm_folder_path = "system_data/hierarchical modulation/two_layers_data"

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

                        if self.show_fit_plot:
                            # Generate fitting curves
                            snr_fit = np.linspace(np.min(self.hm_snr_list.ravel()), np.max(self.hm_snr_list.ravel()),
                                                  100)
                            ber_fit_1 = self.hm_polynomial_model_1(snr_fit)
                            ber_fit_2 = self.hm_polynomial_model_2(snr_fit)

                            # Validate the fitting results and plot
                            plt.scatter(self.hm_snr_list.ravel(), self.hm_ber_1, color='red',
                                        label='original layer 1 BER')
                            plt.scatter(self.hm_snr_list.ravel(), self.hm_ber_2, color='green',
                                        label='original layer 2 BER')
                            plt.plot(snr_fit, ber_fit_1, label='fitting curve for layer 1 BER', color='blue')
                            plt.plot(snr_fit, ber_fit_2, label='fitting curve for layer 2 BER', color='blue')
                            plt.xlabel('SNR (dB)')
                            plt.ylabel('BER')
                            plt.title('SNR vs BER fitting (HM)')
                            plt.legend()
                            plt.grid(True)
                            plt.show()

        self.tm_folder_path = "system_data/typical modulation"
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

                        if self.show_fit_plot:
                            # Generate fitted curve
                            snr_fit = np.linspace(np.min(self.tm_snr_list.ravel()), np.max(self.tm_snr_list.ravel()),
                                                  100)
                            ber_fit = self.tm_polynomial_model(snr_fit)

                            # Plot the original data and the fitted curve
                            plt.scatter(self.tm_snr_list.ravel(), self.tm_ber, color='red', label='Original Data')
                            plt.plot(snr_fit, ber_fit, label=f'Fitted Curve (degree={self.tm_degree})', color='blue')
                            plt.xlabel('SNR (dB)')
                            plt.ylabel('BER')
                            plt.title('SNR vs BER Fitting (TM)')
                            plt.legend()
                            plt.grid(True)
                            plt.show()

                    else:
                        continue

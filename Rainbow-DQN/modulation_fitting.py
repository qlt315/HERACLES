import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import os
import re
from scipy.io import loadmat, savemat
import envs.utils_proposed as util
import random
import matplotlib.pyplot as plt


class EnvProposed(gym.Env):
    def __init__(self):
        self.name = "proposed"
        # Parameter settings
        self.est_err_para = 0.5  # Channel estimation error parameter
        self.hm_power_ratio = 0.9  # Choose from 0.5-0.9



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

                        # Generate fitting curves
                        hm_snr_fit = np.linspace(np.min(self.hm_snr_list.ravel()), np.max(self.hm_snr_list.ravel()),
                                              100)
                        ber_fit_1 = self.hm_polynomial_model_1(hm_snr_fit)
                        ber_fit_2 = self.hm_polynomial_model_2(hm_snr_fit)


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
                        self.tm_ber_qpsk = self.tm_data['bler']
                        print(self.tm_ber_qpsk)
                        self.tm_snr_list_qpsk = np.arange(self.tm_snr_min, self.tm_snr_max + self.tm_snr_int,
                                                          self.tm_snr_int)

                        # Function fitting
                        self.tm_degree = 7
                        self.tm_coefficients = np.polyfit(self.tm_snr_list_qpsk.ravel(), self.tm_ber_qpsk.ravel(),
                                                          self.tm_degree)
                        self.tm_polynomial_model = np.poly1d(self.tm_coefficients)

                        # Generate fitted curve
                        snr_fit_qpsk = np.linspace(np.min(self.tm_snr_list_qpsk.ravel()),
                                                   np.max(self.tm_snr_list_qpsk.ravel()),
                                                   100)
                        ber_fit_qpsk = self.tm_polynomial_model(snr_fit_qpsk)
                        print(ber_fit_qpsk)
        self.tm_folder_path = "system_data/typical modulation"
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
                        self.tm_ber_16qam = self.tm_data['bler']
                        self.tm_snr_list_16qam = np.arange(self.tm_snr_min, self.tm_snr_max + self.tm_snr_int,
                                                           self.tm_snr_int)

                        # Function fitting
                        self.tm_degree = 7
                        self.tm_coefficients = np.polyfit(self.tm_snr_list_16qam.ravel(), self.tm_ber_16qam.ravel(),
                                                          self.tm_degree)
                        self.tm_polynomial_model = np.poly1d(self.tm_coefficients)

                        # Generate fitted curve
                        snr_fit_16qam = np.linspace(np.min(self.tm_snr_list_16qam.ravel()),
                                                    np.max(self.tm_snr_list_16qam.ravel()),
                                                    100)
                        ber_fit_16qam = self.tm_polynomial_model(snr_fit_16qam)

        self.tm_folder_path = "system_data/typical modulation"
        tm_pattern = r'snr_([\d.]+)_([\d.]+)_([\d.]+)_(\w+)_esterr_([\d\.]+)_rate_(\d+)_(\d+)'
        self.tm_mod = "64qam"  # qpsk, 16qam, 64qam
        self.tm_coding_rate_num = 2
        self.tm_coding_rate_den = 3
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
                        self.tm_ber_64qam = self.tm_data['bler']
                        self.tm_snr_list_64qam = np.arange(self.tm_snr_min, self.tm_snr_max + self.tm_snr_int,
                                                           self.tm_snr_int)

                        # Function fitting
                        self.tm_degree = 7
                        self.tm_coefficients = np.polyfit(self.tm_snr_list_64qam.ravel(), self.tm_ber_64qam.ravel(),
                                                          self.tm_degree)
                        self.tm_polynomial_model = np.poly1d(self.tm_coefficients)

                        # Generate fitted curve
                        snr_fit_64qam = np.linspace(np.min(self.tm_snr_list_64qam.ravel()),
                                                    np.max(self.tm_snr_list_64qam.ravel()),
                                                    100)
                        ber_fit_64qam = self.tm_polynomial_model(snr_fit_64qam)

        mat_name = "system_data/modulation_fitting_data.mat"
        savemat(mat_name,
                {"hm_snr_list": self.hm_snr_list,
                 "hm_ber_1": self.hm_ber_1,
                 "hm_ber_2": self.hm_ber_2,
                 "hm_snr_fit_list": hm_snr_fit,
                 "hm_ber_1_fit": ber_fit_1,
                 "hm_ber_2_fit": ber_fit_2,

                 "tm_snr_list_qpsk": self.tm_snr_list_qpsk,
                 "tm_snr_fit_list_qpsk": snr_fit_qpsk,
                 "tm_ber_qpsk": self.tm_ber_qpsk,
                 "tm_ber_qpsk_fit": ber_fit_qpsk,

                 "tm_snr_list_16qam": self.tm_snr_list_16qam,
                 "tm_snr_fit_list_16qam": snr_fit_16qam,
                 "tm_ber_16qam": self.tm_ber_16qam,
                 "tm_ber_16qam_fit": ber_fit_16qam,

                 "tm_snr_list_64qam": self.tm_snr_list_64qam,
                 "tm_snr_fit_list_64qam": snr_fit_64qam,
                 "tm_ber_64qam": self.tm_ber_64qam,
                 "tm_ber_64qam_fit": ber_fit_64qam
                 })


env = EnvProposed()

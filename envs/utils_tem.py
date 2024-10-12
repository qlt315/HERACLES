import numpy as np
from scipy.io import loadmat
import os
import pandas as pd

def per_list_gen(tm_ber, data_size, tm_rate):
    camera_per = 1 - (1 - tm_ber) ** (
            data_size[0, 0] / tm_rate)  # data_size[1] = data_size[2] as we have 2 same cameras
    radar_per = 1 - (1 - tm_ber) ** (data_size[0, 2] / tm_rate)
    lidar_per = 1 - (1 - tm_ber) ** (data_size[0, 3] / tm_rate)
    dual_camera_per = 1 - ((1 - tm_ber) ** (2 * data_size[0, 0]))
    radar_lidar_per = 1 - ((1 - tm_ber) ** ((data_size[0, 2] + data_size[0, 3]) / tm_rate))
    camera_lidar_per = 1 - ((1 - tm_ber) ** ((data_size[0, 0] + data_size[0, 3]) / tm_rate))
    radar_lidar_fusion_radar_per = radar_lidar_per
    dual_camera_fusion_camera_per = dual_camera_per
    camera_lidar_fusion_lidar_per = camera_lidar_per

    per_list = [camera_per.item(), radar_per.item(), lidar_per.item(), dual_camera_per.item(), radar_lidar_per.item(),
                camera_lidar_per.item(),
                camera_per.item(), radar_per.item(), lidar_per.item(), dual_camera_per.item(), radar_lidar_per.item(),
                camera_lidar_per.item(),
                camera_per.item(), radar_per.item(), lidar_per.item(), dual_camera_per.item(), radar_lidar_per.item(),
                camera_lidar_per.item(),
                radar_lidar_fusion_radar_per.item(), dual_camera_fusion_camera_per.item(),
                camera_lidar_fusion_lidar_per.item()
                ]
    per_list = np.array(per_list)
    # print("HM BER 1:", hm_ber_1)
    # print("HM BER 2:", hm_ber_2)
    # print("TM BER:", tm_ber)
    # print(per_list)
    assert not np.any(per_list < 0), "PER CALCULATION ERROR!"
    return per_list

def acc_normalize(acc_exp, current_context):
    platform_data = loadmat("system_data/platform_data.mat")
    acc_sunny_list = platform_data["sunny"][:21]
    acc_rain_list = platform_data["rain"][:21]
    acc_snow_list = platform_data["snow"][:21]
    acc_motorway_list = platform_data["motorway"][:21]
    acc_fog_list = platform_data["fog"][:21]
    acc_night_list = platform_data["night"][:21]

    # print(np.max(acc_sunny_list),np.max(acc_rain_list),np.max(acc_snow_list),
    #       np.max(acc_motorway_list),np.max(acc_fog_list),np.max(acc_night_list))

    try:
        if current_context == "sunny":
            acc_exp_nor = acc_exp / np.max(acc_sunny_list)
        elif current_context == "rain":
            acc_exp_nor = acc_exp / np.max(acc_rain_list)
        elif current_context == "snow":
            acc_exp_nor = acc_exp / np.max(acc_snow_list)
        elif current_context == "motorway":
            acc_exp_nor = acc_exp / np.max(acc_motorway_list)
        elif current_context == "fog":
            acc_exp_nor = acc_exp / np.max(acc_fog_list)
        elif current_context == "night":
            acc_exp_nor = acc_exp / np.max(acc_night_list)
    except:
        print("Acc exp calculation failed!")
    return acc_exp_nor

def normalize_list(input_list):
    inv_list = 1 / (input_list + 1e-10)
    return inv_list / np.sum(inv_list)


def acc_exp_gen(per_list, current_context):
    platform_data = loadmat("system_data/platform_data.mat")
    acc_sunny_list = platform_data["sunny"][:21]
    acc_rain_list = platform_data["rain"][:21]
    acc_snow_list = platform_data["snow"][:21]
    acc_motorway_list = platform_data["motorway"][:21]
    acc_fog_list = platform_data["fog"][:21]
    acc_night_list = platform_data["night"][:21]
    normalized_per_list = normalize_list(per_list)
    acc_exp = 0
    try:
        if current_context == "sunny":
            acc_exp = np.sum(np.multiply(normalized_per_list, acc_sunny_list.T))
        elif current_context == "rain":
            acc_exp = np.sum(np.multiply(normalized_per_list, acc_rain_list.T))
        elif current_context == "snow":
            acc_exp = np.sum(np.multiply(normalized_per_list, acc_snow_list.T))
        elif current_context == "motorway":
            acc_exp = np.sum(np.multiply(normalized_per_list, acc_motorway_list.T))
        elif current_context == "fog":
            acc_exp = np.sum(np.multiply(normalized_per_list, acc_fog_list.T))
        elif current_context == "night":
            acc_exp = np.sum(np.multiply(normalized_per_list, acc_night_list.T))
    except:
        print("Acc exp calculation failed!")

    return acc_exp


def action_gen():
    class ActionDic:
        def __init__(self, id):
            self.id = id
            self.fusion_name = None  # Stem and branch to use
            self.mod_type = None  # Modulation type -- HM or TM
            self.backbone = None
            self.acc = None
            self.com_delay = None  # Computing delay
            self.com_energy = None  # Computing energy consumption

    # Adding action manually :)
    action_sunny_list = [ActionDic(i) for i in range(1, 22)]  # 21 actions for each context
    action_rain_list = [ActionDic(i) for i in range(1, 22)]
    action_snow_list = [ActionDic(i) for i in range(1, 22)]
    action_motorway_list = [ActionDic(i) for i in range(1, 22)]
    action_fog_list = [ActionDic(i) for i in range(1, 22)]
    action_night_list = [ActionDic(i) for i in range(1, 22)]

    #  1,2--Camera 3--Radar 4--Lidar
    #  Array order represents modulation order
    fusion_name_list = [[1], [3], [4], [1, 2], [3, 4], [1, 4],
                        [1], [3], [4], [1, 2], [3, 4], [1, 4],
                        [1], [3], [4], [1, 2], [3, 4], [1, 4],
                        [3, 4], [1, 2], [1, 4]]

    mod_type_list = ["TM"] * 21

    backbone_list = ["18", "18", "18", "18", "18", "18",
                     "50", "50", "50", "50", "50", "50",
                     "101", "101", "101", "101", "101", "101",
                     "18", "18", "18", ]

    platform_data = loadmat("system_data/platform_data.mat")
    acc_sunny_list = platform_data["sunny"][:21]
    acc_rain_list = platform_data["rain"][:21]
    acc_snow_list = platform_data["snow"][:21]
    acc_motorway_list = platform_data["motorway"][:21]
    acc_fog_list = platform_data["fog"][:21]
    acc_night_list = platform_data["night"][:21]

    stem_camera_delay = platform_data["stem_delay"][0]
    stem_radar_delay = platform_data["stem_delay"][2]
    stem_lidar_delay = platform_data["stem_delay"][3]
    stem_dual_camera_delay = stem_camera_delay * 2
    stem_radar_lidar_delay = stem_radar_delay + stem_lidar_delay
    stem_camera_lidar_delay = stem_camera_delay + stem_lidar_delay
    stem_radar_lidar_fusion_radar_delay = stem_radar_lidar_delay
    stem_dual_camera_fusion_camera_delay = stem_dual_camera_delay
    stem_camera_lidar_fusion_lidar_delay = stem_camera_lidar_delay
    stem_delay_list = [stem_camera_delay, stem_radar_delay, stem_lidar_delay, stem_dual_camera_delay,
                       stem_radar_lidar_delay, stem_camera_lidar_delay,
                       stem_camera_delay, stem_radar_delay, stem_lidar_delay, stem_dual_camera_delay,
                       stem_radar_lidar_delay, stem_camera_lidar_delay,
                       stem_camera_delay, stem_radar_delay, stem_lidar_delay, stem_dual_camera_delay,
                       stem_radar_lidar_delay, stem_camera_lidar_delay,
                       stem_radar_lidar_fusion_radar_delay, stem_dual_camera_fusion_camera_delay,
                       stem_camera_lidar_fusion_lidar_delay]

    stem_camera_energy = platform_data["stem_energy"][0]
    stem_radar_energy = platform_data["stem_energy"][2]
    stem_lidar_energy = platform_data["stem_energy"][3]
    stem_dual_camera_energy = stem_camera_energy * 2
    stem_radar_lidar_energy = stem_radar_energy + stem_lidar_energy
    stem_camera_lidar_energy = stem_camera_energy + stem_lidar_energy
    stem_radar_lidar_fusion_radar_energy = stem_radar_lidar_energy
    stem_dual_camera_fusion_camera_energy = stem_dual_camera_energy
    stem_camera_lidar_fusion_lidar_energy = stem_camera_lidar_energy
    stem_energy_list = [stem_camera_energy, stem_radar_energy, stem_lidar_energy, stem_dual_camera_energy,
                        stem_radar_lidar_energy, stem_camera_lidar_energy,
                        stem_camera_energy, stem_radar_energy, stem_lidar_energy, stem_dual_camera_energy,
                        stem_radar_lidar_energy, stem_camera_lidar_energy,
                        stem_camera_energy, stem_radar_energy, stem_lidar_energy, stem_dual_camera_energy,
                        stem_radar_lidar_energy, stem_camera_lidar_energy,
                        stem_radar_lidar_fusion_radar_energy, stem_dual_camera_fusion_camera_energy,
                        stem_camera_lidar_fusion_lidar_energy]

    branch_delay_list = platform_data["branch_delay"][:21]
    branch_energy_list = platform_data["branch_energy"][:21]

    for i in range(len(action_sunny_list)):
        action_sunny_list[i].fusion_name = fusion_name_list[i]
        action_rain_list[i].fusion_name = fusion_name_list[i]
        action_snow_list[i].fusion_name = fusion_name_list[i]
        action_motorway_list[i].fusion_name = fusion_name_list[i]
        action_fog_list[i].fusion_name = fusion_name_list[i]
        action_night_list[i].fusion_name = fusion_name_list[i]

        action_sunny_list[i].mod_type = mod_type_list[i]
        action_rain_list[i].mod_type = mod_type_list[i]
        action_snow_list[i].mod_type = mod_type_list[i]
        action_motorway_list[i].mod_type = mod_type_list[i]
        action_fog_list[i].mod_type = mod_type_list[i]
        action_night_list[i].mod_type = mod_type_list[i]

        action_sunny_list[i].backbone = backbone_list[i]
        action_rain_list[i].backbone = backbone_list[i]
        action_snow_list[i].backbone = backbone_list[i]
        action_motorway_list[i].backbone = backbone_list[i]
        action_fog_list[i].backbone = backbone_list[i]
        action_night_list[i].backbone = backbone_list[i]

        action_sunny_list[i].acc = acc_sunny_list[i]
        action_rain_list[i].acc = acc_rain_list[i]
        action_snow_list[i].acc = acc_snow_list[i]
        action_motorway_list[i].acc = acc_motorway_list[i]
        action_fog_list[i].acc = acc_fog_list[i]
        action_night_list[i].acc = acc_night_list[i]

        action_sunny_list[i].com_delay = branch_delay_list[i] + stem_delay_list[i]
        action_rain_list[i].com_delay = branch_delay_list[i] + stem_delay_list[i]
        action_snow_list[i].com_delay = branch_delay_list[i] + stem_delay_list[i]
        action_motorway_list[i].com_delay = branch_delay_list[i] + stem_delay_list[i]
        action_fog_list[i].com_delay = branch_delay_list[i] + stem_delay_list[i]
        action_night_list[i].com_delay = branch_delay_list[i] + stem_delay_list[i]

        action_sunny_list[i].com_energy = branch_energy_list[i] + stem_energy_list[i]
        action_rain_list[i].com_energy = branch_energy_list[i] + stem_energy_list[i]
        action_snow_list[i].com_energy = branch_energy_list[i] + stem_energy_list[i]
        action_motorway_list[i].com_energy = branch_energy_list[i] + stem_energy_list[i]
        action_fog_list[i].com_energy = branch_energy_list[i] + stem_energy_list[i]
        action_night_list[i].com_energy = branch_energy_list[i] + stem_energy_list[i]

    return (action_sunny_list, action_rain_list, action_snow_list,
            action_motorway_list, action_fog_list, action_night_list)


def action_mapping(action_sunny_list, action_rain_list, action_snow_list,
                   action_motorway_list, action_fog_list, action_night_list, current_context, action):
    try:
        if current_context == "sunny":
            action_info = action_sunny_list[action]
        elif current_context == "rain":
            action_info = action_rain_list[action]
        elif current_context == "snow":
            action_info = action_snow_list[action]
        elif current_context == "motorway":
            action_info = action_motorway_list[action]
        elif current_context == "fog":
            action_info = action_fog_list[action]
        elif current_context == "night":
            action_info = action_night_list[action]
    except:
        print("Action mapping failed!")
        action_info = None

    return action_info


def obtain_cqi_and_snr(directory_path, slot_num):

    # Lists to store SNR and CQI data separately
    snr_list = []
    cqi_list = []

    # Loop through all files in the given directory
    for filename in os.listdir(directory_path):
        # Check if the file is an Excel file
        if filename.endswith(".csv") or filename.endswith(".xls"):
            file_path = os.path.join(directory_path, filename)
            # Read the Excel file and extract the "SNR" and "CQI" columns
            df = pd.read_csv(file_path, usecols=["SNR", "CQI"])

            # Append the "SNR" and "CQI" values to their respective lists
            snr_list.extend(df["SNR"].values)
            cqi_list.extend(df["CQI"].values)

    # Convert the lists into 1D NumPy arrays
    snr_array = np.array(snr_list)
    cqi_array = np.array(cqi_list)

    snr_array = snr_array[snr_array != '-']
    cqi_array = cqi_array[cqi_array != '-']

    snr_array = np.random.choice(snr_array, size=slot_num, replace=False)
    cqi_array = np.random.choice(cqi_array, size=slot_num, replace=False)
    return snr_array, cqi_array


import numpy as np


def estimate_cqi(cqi_true, est_err_para, min_cqi=1, max_cqi=15):
    """
    Estimate the CQI value based on true CQI and a noise-controlled parameter.

    Args:
    - cqi_true (int): The true CQI value.
    - sigma (float): Standard deviation of the noise; higher means less accurate estimation.
    - min_cqi (int): Minimum valid CQI value. Default is 1.
    - max_cqi (int): Maximum valid CQI value. Default is 15.

    Returns:
    - cqi_estimated (int): The estimated CQI value with noise added.
    """
    # Add Gaussian noise to the true CQI value
    noise = np.random.normal(0, est_err_para)
    # Estimate CQI by adding noise and rounding to the nearest integer
    cqi_estimated = np.round(cqi_true + noise)

    # Ensure the estimated CQI is within the valid range [min_cqi, max_cqi]
    cqi_estimated = np.clip(cqi_estimated, min_cqi, max_cqi)

    return int(cqi_estimated)

def obtain_min_acc(current_context):
    platform_data = loadmat("system_data/platform_data.mat")
    acc_sunny_list = platform_data["sunny"][:21]
    acc_rain_list = platform_data["rain"][:21]
    acc_snow_list = platform_data["snow"][:21]
    acc_motorway_list = platform_data["motorway"][:21]
    acc_fog_list = platform_data["fog"][:21]
    acc_night_list = platform_data["night"][:21]
    try:
        if current_context == "sunny":
            min_acc = np.random.uniform(low=np.min(acc_sunny_list), high=np.max(acc_sunny_list), size=1)
        elif current_context == "rain":
            min_acc = np.random.uniform(low=np.min(acc_rain_list), high=np.max(acc_rain_list), size=1)
        elif current_context == "snow":
            min_acc = np.random.uniform(low=np.min(acc_snow_list), high=np.max(acc_snow_list), size=1)
        elif current_context == "motorway":
            min_acc = np.random.uniform(low=np.min(acc_motorway_list), high=np.max(acc_motorway_list), size=1)
        elif current_context == "fog":
            min_acc = np.random.uniform(low=np.min(acc_fog_list), high=np.max(acc_fog_list), size=1)
        elif current_context == "night":
            min_acc = np.random.uniform(low=np.min(acc_night_list), high=np.max(acc_night_list), size=1)
    except:
        print("min acc calcualtion failed!")

    return min_acc / 150
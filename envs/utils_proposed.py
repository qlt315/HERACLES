import numpy as np
import os
import pandas as pd
from scipy.io import loadmat

def normalize_list(input_list):
    inv_list = 1 / (input_list + 1e-10)
    return inv_list / np.sum(inv_list)

def acc_normalize(acc_exp, current_context):
    platform_data = loadmat("/home/ababu/HERACLES/system_data/platform_data.mat")
    acc_sunny_list = platform_data["sunny"][:21]
    acc_rain_list = platform_data["rain"][:21]
    acc_snow_list = platform_data["snow"][:21]
    acc_motorway_list = platform_data["motorway"][:21]
    acc_fog_list = platform_data["fog"][:21]
    acc_night_list = platform_data["night"][:21]
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
        acc_exp_nor = acc_exp
    return acc_exp_nor

def action_list_gen(acc_list):
    a = acc_list
    return [a[0], a[1], a[2], a[3], a[3], a[4], a[4], a[5], a[5],
            a[6], a[7], a[8], a[9], a[9], a[10], a[10], a[11], a[11],
            a[12], a[13], a[14], a[15], a[15], a[16], a[16], a[17], a[17],
            a[18], a[18], a[19], a[19], a[20], a[20]]

def action_gen():
    class ActionDic:
        def __init__(self, id):
            self.id = id
            self.fusion_name = None
            self.backbone = None
            self.acc = None
            self.com_delay = None
            self.com_energy = None

    action_sunny_list = [ActionDic(i) for i in range(1, 34)]
    action_rain_list = [ActionDic(i) for i in range(1, 34)]
    action_snow_list = [ActionDic(i) for i in range(1, 34)]
    action_motorway_list = [ActionDic(i) for i in range(1, 34)]
    action_fog_list = [ActionDic(i) for i in range(1, 34)]
    action_night_list = [ActionDic(i) for i in range(1, 34)]

    fusion_name_list = [[1], [3], [4], [1, 2], [2, 1], [3, 4], [4, 3], [1, 4], [4, 1],
                        [1], [3], [4], [1, 2], [2, 1], [3, 4], [4, 3], [1, 4], [4, 1],
                        [1], [3], [4], [1, 2], [2, 1], [3, 4], [4, 3], [1, 4], [4, 1],
                        [3, 4], [4, 3], [1, 2], [2, 1], [1, 4], [4, 1]]

    backbone_list = ["18"] * 9 + ["50"] * 9 + ["101"] * 9 + ["18"] * 6

    platform_data = loadmat("system_data/platform_data.mat")
    acc_sunny_list = action_list_gen(platform_data["sunny"][:21])
    acc_rain_list = action_list_gen(platform_data["rain"][:21])
    acc_snow_list = action_list_gen(platform_data["snow"][:21])
    acc_motorway_list = action_list_gen(platform_data["motorway"][:21])
    acc_fog_list = action_list_gen(platform_data["fog"][:21])
    acc_night_list = action_list_gen(platform_data["night"][:21])

    stem_delay = platform_data["stem_delay"]
    stem_energy = platform_data["stem_energy"]

    stem_delay_list = [stem_delay[0], stem_delay[2], stem_delay[3], stem_delay[0]*2,
                       stem_delay[0]*2, stem_delay[2]+stem_delay[3], stem_delay[2]+stem_delay[3],
                       stem_delay[0]+stem_delay[3], stem_delay[0]+stem_delay[3]] * 3 + \
                      [stem_delay[2]+stem_delay[3]] * 2 + [stem_delay[0]*2] * 2 + [stem_delay[0]+stem_delay[3]] * 2

    stem_energy_list = [stem_energy[0], stem_energy[2], stem_energy[3], stem_energy[0]*2,
                        stem_energy[0]*2, stem_energy[2]+stem_energy[3], stem_energy[2]+stem_energy[3],
                        stem_energy[0]+stem_energy[3], stem_energy[0]+stem_energy[3]] * 3 + \
                       [stem_energy[2]+stem_energy[3]] * 2 + [stem_energy[0]*2] * 2 + [stem_energy[0]+stem_energy[3]] * 2

    branch_delay_list = action_list_gen(platform_data["branch_delay"][:21])
    branch_energy_list = action_list_gen(platform_data["branch_energy"][:21])

    for i in range(len(action_sunny_list)):
        for context_list, acc_list in zip(
            [action_sunny_list, action_rain_list, action_snow_list,
             action_motorway_list, action_fog_list, action_night_list],
            [acc_sunny_list, acc_rain_list, acc_snow_list,
             acc_motorway_list, acc_fog_list, acc_night_list]):
            context_list[i].fusion_name = fusion_name_list[i]
            context_list[i].backbone = backbone_list[i]
            context_list[i].acc = acc_list[i]
            context_list[i].com_delay = branch_delay_list[i] + stem_delay_list[i]
            context_list[i].com_energy = branch_energy_list[i] + stem_energy_list[i]

    return (action_sunny_list, action_rain_list, action_snow_list,
            action_motorway_list, action_fog_list, action_night_list)

def action_mapping(action_sunny_list, action_rain_list, action_snow_list,
                   action_motorway_list, action_fog_list, action_night_list, current_context, action):
    try:
        if current_context == "sunny":
            return action_sunny_list[action]
        elif current_context == "rain":
            return action_rain_list[action]
        elif current_context == "snow":
            return action_snow_list[action]
        elif current_context == "motorway":
            return action_motorway_list[action]
        elif current_context == "fog":
            return action_fog_list[action]
        elif current_context == "night":
            return action_night_list[action]
    except:
        print("Action mapping failed!")
        return None

def estimate_cqi(cqi_true, est_err_para, min_cqi=1, max_cqi=15):
    noise = np.random.normal(0, est_err_para)
    cqi_estimated = np.round(cqi_true + noise)
    return int(np.clip(cqi_estimated, min_cqi, max_cqi))

def obtain_min_acc(current_context):
    platform_data = loadmat("/home/ababu/HERACLES/system_data/platform_data.mat")
    try:
        acc_list = platform_data[current_context][:21]
        min_acc = np.random.uniform(low=np.min(acc_list), high=np.max(acc_list), size=1)
    except:
        print("min acc calculation failed!")
        min_acc = np.array([0.1])
    return min_acc / 125

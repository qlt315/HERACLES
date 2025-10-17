# Collect the information of each action value
import os
import pandas as pd
import numpy as np
from itertools import permutations
from scipy.io import loadmat


def acc_exp_gen(per, current_context):
    platform_data = loadmat("/home/ababu/HERACLES/system_data/platform_data.mat")
    acc_context = platform_data.get(current_context, [0])[-1]
    try:
        acc_exp = acc_context * (1 - per)
    except:
        print("Acc exp calculation failed!")
        acc_exp = 0
    return acc_exp

def acc_normalize(acc_exp, current_context):
    platform_data = loadmat("/home/ababu/HERACLES/system_data/platform_data.mat")
    try:
        acc_list = platform_data[current_context][:21]
        acc_exp_nor = acc_exp / np.max(acc_list)
    except:
        print("Acc exp normalization failed!")
        acc_exp_nor = acc_exp
    return acc_exp_nor

def action_gen():
    class ActionDic:
        def __init__(self, id):
            self.id = id
            self.fusion_name = None
            self.backbone = None
            self.acc = None
            self.com_delay = None
            self.com_energy = None

    action_sunny_list = [ActionDic(i) for i in range(1, 25)]
    action_rain_list = [ActionDic(i) for i in range(1, 25)]
    action_snow_list = [ActionDic(i) for i in range(1, 25)]
    action_motorway_list = [ActionDic(i) for i in range(1, 25)]
    action_fog_list = [ActionDic(i) for i in range(1, 25)]
    action_night_list = [ActionDic(i) for i in range(1, 25)]

    fusion_name_list = [list(p) for p in permutations([1, 2, 3, 4])]
    backbone_list = ["18"] * 24

    platform_data = loadmat("/home/ababu/HERACLES/system_data/platform_data.mat")
    acc_contexts = {
        "sunny": platform_data["sunny"][-1],
        "rain": platform_data["rain"][-1],
        "snow": platform_data["snow"][-1],
        "motorway": platform_data["motorway"][-1],
        "fog": platform_data["fog"][-1],
        "night": platform_data["night"][-1]
    }

    stem_delay = platform_data["stem_delay"]
    stem_energy = platform_data["stem_energy"]
    branch_delay = platform_data["branch_delay"][-1]
    branch_energy = platform_data["branch_energy"][-1]

    stem_total_delay = stem_delay[0]*2 + stem_delay[2] + stem_delay[3]
    stem_total_energy = stem_energy[0]*2 + stem_energy[2] + stem_energy[3]

    for i in range(24):
        for context, action_list in zip(
            acc_contexts.keys(),
            [action_sunny_list, action_rain_list, action_snow_list,
             action_motorway_list, action_fog_list, action_night_list]):
            action_list[i].fusion_name = fusion_name_list[i]
            action_list[i].backbone = backbone_list[i]
            action_list[i].acc = acc_contexts[context]
            action_list[i].com_delay = branch_delay + stem_total_delay
            action_list[i].com_energy = branch_energy + stem_total_energy

    return (action_sunny_list, action_rain_list, action_snow_list,
            action_motorway_list, action_fog_list, action_night_list)

def action_mapping(action_sunny_list, action_rain_list, action_snow_list,
                   action_motorway_list, action_fog_list, action_night_list, current_context, action):
    try:
        context_map = {
            "sunny": action_sunny_list,
            "rain": action_rain_list,
            "snow": action_snow_list,
            "motorway": action_motorway_list,
            "fog": action_fog_list,
            "night": action_night_list
        }
        return context_map[current_context][action]
    except (IndexError, KeyError) as e:
        print("Action mapping failed:", e)
        return None

def obtain_cqi_and_snr(directory_path, slot_num):
    snr_list = []
    cqi_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv") or filename.endswith(".xls"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path, usecols=["SNR", "CQI"])
            snr_list.extend(df["SNR"].values)
            cqi_list.extend(df["CQI"].values)
    snr_array = np.array(snr_list)
    cqi_array = np.array(cqi_list)
    snr_array = snr_array[snr_array != '-']
    cqi_array = cqi_array[cqi_array != '-']
    snr_array = np.random.choice(snr_array, size=slot_num, replace=False)
    cqi_array = np.random.choice(cqi_array, size=slot_num, replace=False)
    return snr_array, cqi_array

def estimate_cqi(cqi_true, est_err_para, min_cqi=1, max_cqi=15):
    noise = np.random.normal(0, est_err_para)
    cqi_estimated = np.round(cqi_true + noise)
    return int(np.clip(cqi_estimated, min_cqi, max_cqi))
def obtain_min_acc(current_context):
    platform_data = loadmat("/home/ababu/HERACLES/system_data/platform_data.mat")
    try:
        acc_context_map = {
            "sunny": platform_data["sunny"][:21],
            "rain": platform_data["rain"][:21],
            "snow": platform_data["snow"][:21],
            "motorway": platform_data["motorway"][:21],
            "fog": platform_data["fog"][:21],
            "night": platform_data["night"][:21]
        }
        acc_list = acc_context_map.get(current_context, np.array([0.1]))
        min_acc = np.random.uniform(low=np.min(acc_list), high=np.max(acc_list), size=1)
    except Exception as e:
        print("min acc calculation failed:", e)
        min_acc = np.array([0.1])
    return min_acc / 125

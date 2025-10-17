import os
import pandas as pd
import numpy as np
from numpy.ma.core import shape
from scipy.io import loadmat


def normalize_list(input_list):
    inv_list = 1 / (input_list + 1e-10)
    return inv_list / np.sum(inv_list)

def acc_normalize(acc_exp, current_context):
    platform_data = loadmat("/home/ababu/HERACLES/system_data/platform_data.mat")
    try:
        acc_list = platform_data[current_context][:21]
        acc_exp_nor = acc_exp / np.max(acc_list)
    except:
        print("Acc exp normalization failed!")
        acc_exp_nor = acc_exp
    return acc_exp_nor

def acc_exp_gen(per_list, current_context):
    platform_data = loadmat("/home/ababu/HERACLES/system_data/platform_data.mat")
    try:
        acc_list = platform_data[current_context][:21]
        normalized_per_list = normalize_list(per_list)
        acc_exp = np.sum(np.multiply(normalized_per_list, acc_list.T))
    except:
        print("Acc exp calculation failed!")
        acc_exp = 0
    return acc_exp
from scipy.io import loadmat

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

    fusion_name_list = [[1], [3], [4], [1, 2], [2, 1], [3, 4], [4, 3], [1, 4], [4, 1]] * 3 + \
                       [[3, 4], [4, 3], [1, 2], [2, 1], [1, 4], [4, 1]]

    backbone_list = ["18"] * 9 + ["50"] * 9 + ["101"] * 9 + ["18"] * 6

    def action_list_gen(acc_list):
        return [acc_list[i] if i < len(acc_list) else acc_list[-1] for i in
                [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 9, 10, 10, 11, 11,
                 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20]]

    platform_data = loadmat("/home/ababu/HERACLES/system_data/platform_data.mat")
    acc_contexts = {
        "sunny": action_list_gen(platform_data["sunny"][:21]),
        "rain": action_list_gen(platform_data["rain"][:21]),
        "snow": action_list_gen(platform_data["snow"][:21]),
        "motorway": action_list_gen(platform_data["motorway"][:21]),
        "fog": action_list_gen(platform_data["fog"][:21]),
        "night": action_list_gen(platform_data["night"][:21])
    }

    stem_delay = platform_data["stem_delay"]
    stem_energy = platform_data["stem_energy"]
    branch_delay = action_list_gen(platform_data["branch_delay"][:21])
    branch_energy = action_list_gen(platform_data["branch_energy"][:21])

    stem_delay_list = [stem_delay[0], stem_delay[2], stem_delay[3], stem_delay[0]*2, stem_delay[0]*2,
                       stem_delay[2]+stem_delay[3], stem_delay[2]+stem_delay[3],
                       stem_delay[0]+stem_delay[3], stem_delay[0]+stem_delay[3]] * 3 + \
                      [stem_delay[2]+stem_delay[3]] * 2 + [stem_delay[0]*2] * 2 + [stem_delay[0]+stem_delay[3]] * 2

    stem_energy_list = [stem_energy[0], stem_energy[2], stem_energy[3], stem_energy[0]*2, stem_energy[0]*2,
                        stem_energy[2]+stem_energy[3], stem_energy[2]+stem_energy[3],
                        stem_energy[0]+stem_energy[3], stem_energy[0]+stem_energy[3]] * 3 + \
                       [stem_energy[2]+stem_energy[3]] * 2 + [stem_energy[0]*2] * 2 + [stem_energy[0]+stem_energy[3]] * 2

    for i in range(33):
        for context, action_list in zip(
            acc_contexts.keys(),
            [action_sunny_list, action_rain_list, action_snow_list,
             action_motorway_list, action_fog_list, action_night_list]):
            action_list[i].fusion_name = fusion_name_list[i]
            action_list[i].backbone = backbone_list[i]
            action_list[i].acc = acc_contexts[context][i]
            action_list[i].com_delay = branch_delay[i] + stem_delay_list[i]
            action_list[i].com_energy = branch_energy[i] + stem_energy_list[i]

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
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat

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
    return snr_array.astype(float), cqi_array.astype(int)

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

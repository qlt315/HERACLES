# Collect the information of each action value
import numpy as np
from itertools import permutations
from scipy.io import loadmat


def per_list_gen(hm_ber_list, data_size, hm_rate):
    all_permutations = list(permutations([0, 1, 2, 3]))
    per_list = []
    for perm in all_permutations:
        per_list.append(1 - ((1 - hm_ber_list[0]) ** (data_size[0, perm[0]] / hm_rate)) * (
                ((1 - hm_ber_list[1]) ** (data_size[0, perm[1]] / hm_rate)) * (
                (1 - hm_ber_list[2]) ** (data_size[0, perm[2]] / hm_rate))) * (
                                (1 - hm_ber_list[3]) ** (data_size[0, perm[3]] / hm_rate)))
    per_list = np.array(per_list)
    # per = np.sum(per_list) / len(per_list)
    assert not np.any(per_list < 0), "PER CALCULATION ERROR!"
    return per_list


def acc_exp_gen(per, current_context):
    platform_data = loadmat("system_data/platform_data.mat")
    acc_sunny = platform_data["sunny"][-1]
    acc_rain = platform_data["rain"][-1]
    acc_snow = platform_data["snow"][-1]
    acc_motorway = platform_data["motorway"][-1]
    acc_fog = platform_data["fog"][-1]
    acc_night = platform_data["night"][-1]

    acc_exp = 0

    try:
        if current_context == "sunny":
            acc_exp = acc_sunny * (1 - per)
        elif current_context == "rain":
            acc_exp = acc_rain * (1 - per)
        elif current_context == "snow":
            acc_exp = acc_snow * (1 - per)
        elif current_context == "motorway":
            acc_exp = acc_motorway * (1 - per)
        elif current_context == "fog":
            acc_exp = acc_fog * (1 - per)
        elif current_context == "night":
            acc_exp = acc_night * (1 - per)
    except:
        print("Acc exp calculation failed!")

    return acc_exp

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
    action_sunny_list = [ActionDic(i) for i in range(1, 25)]  # 4*3*2*1 = 24 actions for each context
    action_rain_list = [ActionDic(i) for i in range(1, 25)]
    action_snow_list = [ActionDic(i) for i in range(1, 25)]
    action_motorway_list = [ActionDic(i) for i in range(1, 25)]
    action_fog_list = [ActionDic(i) for i in range(1, 25)]
    action_night_list = [ActionDic(i) for i in range(1, 25)]

    all_permutations = list(permutations([1, 2, 3, 4]))
    fusion_name_list = [list(p) for p in all_permutations]

    mod_type_list = ["HM"] * 24
    backbone_list = ["18"] * 24

    platform_data = loadmat("system_data/platform_data.mat")
    acc_sunny = platform_data["sunny"][-1]
    acc_rain = platform_data["rain"][-1]
    acc_snow = platform_data["snow"][-1]
    acc_motorway = platform_data["motorway"][-1]
    acc_fog = platform_data["fog"][-1]
    acc_night = platform_data["night"][-1]

    stem_camera_delay = platform_data["stem_delay"][0]
    stem_radar_delay = platform_data["stem_delay"][2]
    stem_lidar_delay = platform_data["stem_delay"][3]
    stem_total_delay = stem_camera_delay * 2 + stem_radar_delay + stem_lidar_delay

    stem_camera_energy = platform_data["stem_energy"][0]
    stem_radar_energy = platform_data["stem_energy"][2]
    stem_lidar_energy = platform_data["stem_energy"][3]
    stem_total_energy = stem_camera_energy * 2 + stem_radar_energy + stem_lidar_energy

    branch_total_delay = platform_data["branch_delay"][-1]

    branch_total_energy = platform_data["branch_energy"][-1]

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

        action_sunny_list[i].acc = acc_sunny
        action_rain_list[i].acc = acc_rain
        action_snow_list[i].acc = acc_snow
        action_motorway_list[i].acc = acc_motorway
        action_fog_list[i].acc = acc_fog
        action_night_list[i].acc = acc_night

        action_sunny_list[i].com_delay = branch_total_delay + stem_total_delay
        action_rain_list[i].com_delay = branch_total_delay + stem_total_delay
        action_snow_list[i].com_delay = branch_total_delay + stem_total_delay
        action_motorway_list[i].com_delay = branch_total_delay + stem_total_delay
        action_fog_list[i].com_delay = branch_total_delay + stem_total_delay
        action_night_list[i].com_delay = branch_total_delay + stem_total_delay

        action_sunny_list[i].com_energy = branch_total_energy + stem_total_energy
        action_rain_list[i].com_energy = branch_total_energy + stem_total_energy
        action_snow_list[i].com_energy = branch_total_energy + stem_total_energy
        action_motorway_list[i].com_energy = branch_total_energy + stem_total_energy
        action_fog_list[i].com_energy = branch_total_energy + stem_total_energy
        action_night_list[i].com_energy = branch_total_energy + stem_total_energy

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
    except (IndexError, KeyError) as e:
        print("Action mapping failed:", e)
        action_info = None

    return action_info

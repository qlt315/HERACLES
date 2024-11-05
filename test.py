import torch
import numpy as np
import gym
from envs.env_proposed_origin import EnvProposed_origin
from envs.env_proposed_erf import EnvProposed_erf
from envs.env_tem import EnvTEM
from envs.env_sse import EnvSSE
from torch.utils.tensorboard import SummaryWriter
from baselines import amac
from rainbow_dqn.rainbow_replay_buffer import *
from rainbow_dqn.rainbow_agent import DQN
import argparse
import random
import tools.saving_loading as sl
import time
import pandas as pd
import os
from experiments.eval_bad_actions import get_top_k_values
from collections import Counter
from scipy.io import savemat
time_start = time.time()

context_num = 3
show_action_num = 3
context_list = ["sunny", "snow", "fog", "motorway", "night", "rain"]
algorithms = ["Proposed_erf", "Proposed_origin", "SSE", "TEM", "DQN", "AMAC"]
columns = ["Reward", "Delay", "Energy", "Accuracy", "Accuracy Vio Rate", "Re-trans Num", "Most Picked " + str(show_action_num) + " Action"]
acc_mean = 0.5
acc_std = 0.1
table_data = np.empty([len(context_list), len(algorithms), len(columns)], dtype='U50')
table_data[0,0,1] = str(acc_mean) + "\u00B1" + str(acc_std)
def create_excel(table_data, file_name):
    # Assume data is a 3D array with shape (6 contexts, 6 algorithms, 7 metrics)
    # 6 contexts: ["snow", "fog", "motorway", "night", "rain", "sunny"]
    # 6 algorithms: each context includes 6 rows of data
    # 7 metrics: ["Reward", "Delay", "Energy", "Accuracy", "Accuracy", "Re-trans", "Actions"]



    # Create an empty DataFrame
    df = pd.DataFrame(columns=["Context", "Algorithm"] + columns)

    # Fill in the data
    for i, context in enumerate(context_list):
        for j, algo in enumerate(algorithms):
            # Get the corresponding data values, assuming data has shape (6, 6, 7)
            row_data = table_data[i, j, :]
            # Create the row
            row = [context, algo] + list(row_data)
            # Add the row to the DataFrame
            df.loc[len(df)] = row


    # Output to an Excel file
    df.to_excel(file_name, index=False)


create_excel(table_data, "experiments/diff_context_data/performance_table_diff_context.xlsx")

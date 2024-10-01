import pandas as pd
import scipy.io as scio
import numpy as np
import hdf5storage

# import data
data = hdf5storage.loadmat('train_data/train_proposed_data.mat')
reward_list = np.array(data['train_step_reward'])
reward_list = np.squeeze(reward_list, axis=2)
print(np.shape(reward_list))

reward_list_processed = np.zeros(np.shape(reward_list))

# data processing
rolling_intv = 5
df = pd.DataFrame(reward_list[0, :])
reward_list_processed[0, :] = list(np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))



# def smooth(data, sm=2):
#     smooth_data = []
#     if sm > 1:
#         for d in data:
#             z = np.ones(len(d))
#             y = np.ones(sm) * 1.0
#             d = np.convolve(y, d, "same") / np.convolve(y, z, "same")
#             smooth_data.append(d)
#     return smooth_data


# reward_list_processed = smooth(reward_list)

# data output
file_name = 'reward_list_processed.mat'
scio.savemat(file_name, {'reward_list_processed': reward_list_processed})


import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import scipy.io
from torch.utils.data import TensorDataset, DataLoader
# https://neurotycho.brain.riken.jp/download/2012/20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6.zip

for i in range(1,64):

    channelA = scipy.io.loadmat(f'rep_2/datasets/ecog/ECoG_ch{i}.mat')

    print(channelA[f'ECoGData_ch{i}'][:, :3])
    data = channelA[f'ECoGData_ch{i}'][:, :1000].squeeze()
    print(data.shape)
    plt.plot(data)

plt.title('ECoG Data unprocessed for all 64 channels')
plt.show()

# 36
channel36 = scipy.io.loadmat(f'rep_2/datasets/ecog/ECoG_ch36.mat')
channel36_array = channel36[f'ECoGData_ch36'][:, :1000].squeeze()
channel36_array = (channel36_array - np.mean(channel36_array)) / np.std(channel36_array)

# 35
channel35 = scipy.io.loadmat(f'rep_2/datasets/ecog/ECoG_ch35.mat')
channel35_array = channel35[f'ECoGData_ch35'][:, :1000].squeeze()
channel35_array = (channel35_array - np.mean(channel35_array)) / np.std(channel35_array)

# 20
channel20 = scipy.io.loadmat(f'rep_2/datasets/ecog/ECoG_ch20.mat')
channel20_array = channel20[f'ECoGData_ch20'][:, :1000].squeeze()
channel20_array = (channel20_array - np.mean(channel20_array)) / np.std(channel20_array)


# 64
channel64 = scipy.io.loadmat(f'rep_2/datasets/ecog/ECoG_ch64.mat')
channel64_array = channel64[f'ECoGData_ch64'][:, :1000].squeeze()
channel64_array = (channel64_array - np.mean(channel64_array)) / np.std(channel64_array)

# 19
channel19 = scipy.io.loadmat(f'rep_2/datasets/ecog/ECoG_ch19.mat')
channel19_array = channel19[f'ECoGData_ch19'][:, :1000].squeeze()
channel19_array = (channel19_array - np.mean(channel19_array)) / np.std(channel19_array)


# 36 and 64 are a lot less correlated with the rest of the channels
plt.plot(channel36_array)
plt.plot(channel35_array)
plt.legend(['channel 36 standardised', 'channel 35 standardised'])
plt.show()

plt.plot(channel35_array)
plt.plot(channel20_array)
plt.legend(['channel 35 standardised', 'channel 20 standardised'])
plt.show()

plt.plot(channel64_array)
plt.plot(channel35_array)
plt.legend(['channel 64 standardised', 'channel 35 standardised'])
plt.show()

plt.plot(channel19_array)
plt.plot(channel20_array)
plt.legend(['channel 19 standardised', 'channel 20 standardised'])
plt.show()

plt.plot(channel19_array)
plt.plot(channel36_array)
plt.legend(['channel 19 standardised', 'channel 36 standardised'])
plt.show()


num_channels = 64
data_list = []

for i in range(1, num_channels + 1):
    channel_data = scipy.io.loadmat(f'rep_2/datasets/ecog/ECoG_ch{i}.mat')
    data = channel_data[f'ECoGData_ch{i}'].squeeze()
    # normalize data
    # normalized_data = (data - np.mean(data)) / np.std(data)
    data_list.append(data)
    print(i)

# Stack all channel data into a single numpy array
all_data = np.stack(data_list, axis=0)

# Calculate the correlation matrix
correlation_matrix = np.corrcoef(all_data)

# Plot the correlation matrix
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation matrix between ECoG channels')
plt.show()







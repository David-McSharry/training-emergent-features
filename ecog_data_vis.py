# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import scipy.io
from torch.utils.data import TensorDataset, DataLoader
# https://neurotycho.brain.riken.jp/download/2012/20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6.zip

for i in range(1,64):

    channelA = scipy.io.loadmat(f'ecog_experiment/data/ecog/ECoG_ch{i}.mat')

    print(channelA[f'ECoGData_ch{i}'][:, :3])
    data = channelA[f'ECoGData_ch{i}'][:, :1000].squeeze()
    print(data.shape)
    plt.plot(data)

plt.title('ECoG Data unprocessed for all 64 channels')
plt.show()

# 36
channel36 = scipy.io.loadmat(f'ecog_experiment/data/ecog/ECoG_ch36.mat')
channel36_array = channel36[f'ECoGData_ch36'][:, :1000].squeeze()
channel36_array = (channel36_array - np.mean(channel36_array)) / np.std(channel36_array)

# 35
channel35 = scipy.io.loadmat(f'ecog_experiment/data/ecog/ECoG_ch35.mat')
channel35_array = channel35[f'ECoGData_ch35'][:, :1000].squeeze()
channel35_array = (channel35_array - np.mean(channel35_array)) / np.std(channel35_array)

# 20
channel20 = scipy.io.loadmat(f'ecog_experiment/data/ecog/ECoG_ch20.mat')
channel20_array = channel20[f'ECoGData_ch20'][:, :1000].squeeze()
channel20_array = (channel20_array - np.mean(channel20_array)) / np.std(channel20_array)

# 64
channel64 = scipy.io.loadmat(f'ecog_experiment/data/ecog/ECoG_ch64.mat')
channel64_array = channel64[f'ECoGData_ch64'][:, :1000].squeeze()
channel64_array = (channel64_array - np.mean(channel64_array)) / np.std(channel64_array)

# 19
channel19 = scipy.io.loadmat(f'ecog_experiment/data/ecog/ECoG_ch19.mat')
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
    channel_data = scipy.io.loadmat(f'ecog_experiment/data/ecog/ECoG_ch{i}.mat')
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
plt.imshow(correlation_matrix, cmap='YlGnBu', interpolation='none')
plt.colorbar()
plt.title('Correlation matrix between ECoG channels')
plt.show()

# get the mean correlation with all the other channels for each channel
mean_correlation = np.mean(correlation_matrix, axis=1)
plt.plot(mean_correlation)
plt.title('Mean correlation with all other channels')
plt.show()

# %%
# get the 20 channels with the lowest mean correlation
lowest_correlation_channels = np.argsort(mean_correlation)[:20]
print(lowest_correlation_channels + 1)




# %%

import pandas as pd
# make df below

data_dict = {}

for i in range(1,65):

    channelA = scipy.io.loadmat(f'ecog_experiment/data/ecog/ECoG_ch{i}.mat')

    data_dict[f'ch{i}'] = (channelA[f"ECoGData_ch{i}"]).squeeze(0)


df = pd.DataFrame(data_dict)

# print various metadata

print("head shows the first 5 rows of the dataframe")
print(df.head())
print("tail shows the last 5 rows of the dataframe")
print(df.tail())
print("info shows the datatypes of each column")
print(df.info())
print("describe shows summary statistics for each column")
print(df.describe())
print("shape shows the dimensions of the dataframe")
print(df.shape)
print("columns shows the column names")
print(df.columns)
print("index shows the index names")
print(df.index)
print("dtypes shows the datatypes of each column")
print(df.dtypes)




# %%

import os
# import ecog_experiment/data/ecog_dataset.pth as dataset
import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from ecog_experiment.utils import fft_check
from ecog_experiment.utils import prepare_ecog_dataset


# imoprt
print(os.getcwd())

config_preprocessing = {
    'dim_reduction': 'PCA',
    'n_components': 10
}

prepare_ecog_dataset(config_preprocessing)

dataset = torch.load('ecog_experiment/data/ecog_dataset.pth')

ch1 = dataset[:, 0]
# to numpy
fft_check(ch1.numpy(), 300)

print(dataset.size())


# %%

import os
# import ecog_experiment/data/ecog_dataset.pth as dataset
import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from ecog_experiment.utils import fft_check
from ecog_experiment.utils import prepare_ecog_dataset


# imoprt
print(os.getcwd())

config_preprocessing = {
    'dim_reduction': 'PCA',
    'n_components': 10
}

prepare_ecog_dataset(config_preprocessing)

dataset = torch.load('ecog_experiment/data/ecog_dataset.pth')

ch1 = dataset[:, 0]
# to numpy
fft_check(ch1.numpy(), 300)


# %%

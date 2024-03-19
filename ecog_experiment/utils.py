import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import scipy.io
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt, decimate
from scipy.fftpack import fft, fftfreq
from sklearn.decomposition import PCA


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def fft_check(data, fs):
    # Perform FFT on your data and plot results to view frequencies present in data
    N = data.shape[0]
    yf = fft(data)
    xf = fftfreq(N, 1/fs)

    plt.figure(figsize=(10,6))
    plt.plot(xf, np.abs(yf))

    plt.title('FFT of data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.xscale('log')
    plt.show()

    return None


# https://neurotycho.brain.riken.jp/download/2012/20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6.zip

# TODO: add some optional data preprocessing to this function 
# band pass filter
# only taking the 20 least correlated channels, 20 is arbitrary
# dim reduction
def prepare_ecog_dataset(config):
    
    num_channels = 64
    data_list = []
    for i in range(1, num_channels + 1):

        channel_data = scipy.io.loadmat(f'ecog_experiment/data/ecog/ECoG_ch{i}.mat')
        data = channel_data[f'ECoGData_ch{i}'].squeeze()

        # high pass filter at 1 Hz
        fs = 1000  # original sampling rate
        cutoff = 1  # cutoff frequency for high pass filter
        data = butter_highpass_filter(data, cutoff, fs)

        # downsample to 300 Hz
        downsample_rate = int(fs / 300)  # calculate downsample rate
        data = decimate(data, downsample_rate)

        # standardise across features. Comes last as the data being fed into our ML model should be standardised
        data = (data - np.mean(data)) / np.std(data)

        data_list.append(data) 

    # Stack all channel data into a single numpy array
    all_data = np.stack(data_list, axis=0)

    dataset_tensor = torch.tensor(all_data).T

    print('before dim reduction', dataset_tensor.shape)

    if config['dim_reduction'] == 'PCA':
        # perform PCA
        pca = PCA(n_components=config['n_components'])
        dataset_tensor = pca.fit_transform(dataset_tensor)

        # tranform back to torch tensor
        dataset_tensor = torch.tensor(dataset_tensor)
    elif config['dim_reduction'] == 'None':
        pass
    else:
        raise ValueError("dim_reduction not specified")

    # save as dataset
    torch.save(dataset_tensor, 'ecog_experiment/data/ecog_dataset.pth')

    return None


def prepare_batch(X):
    # take all samples except the last one as inputs
    input_data = X[:-1]

    # take all samples except the first one as targets
    target_data = X[1:]

    # stack them as pairs with dimension (999, 2, 10)
    pairs = torch.stack((input_data, target_data), dim=1)
    
    assert pairs[0,0,0] == input_data[0,0]

    return pairs




# def check_ecog_dataset():
#     # load the dataset
#     dataset = torch.load('ecog_experiment/data/ecog_dataset.pth')
#     # if not print the mean of the dataset

#     if not torch.allclose(
#         torch.mean(dataset, dim=0).to(torch.float32),
#         torch.zeros(64).to(torch.float32)
#     ):
#         print(torch.mean(dataset, dim=0).to(torch.float32))
#         raise ValueError("The mean of the dataset is not zero")




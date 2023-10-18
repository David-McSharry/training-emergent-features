# %%
from torch.utils.data import Dataset, DataLoader
import torch
from train import train_supervenient_representation_model
loaded_dataset = torch.load('data/bit_string_dataset.pth')

# make dataset into a dataloader

dataloader = DataLoader(loaded_dataset, batch_size=100, shuffle=True)


# %%

# train the model
torch.device('cpu')
device = "cpu"

# train different models for 10 values of clip
import numpy as np
model = train_supervenient_representation_model(device, dataloader, clip=3)

 
# %%



print(model.f_supervenient(torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.float32)))



# %%

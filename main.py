from torch.utils.data import Dataset, DataLoader
import torch
from train import train_supervenient_representation_model
loaded_dataset = torch.load('bit_string_dataset.pth')

# make dataset into a dataloader

dataloader = DataLoader(loaded_dataset, batch_size=3, shuffle=True)

# train the model
torch.device('cpu')
device = "cpu"
model = train_supervenient_representation_model(device, dataloader)

 
  


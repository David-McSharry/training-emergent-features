# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import wandb

from utils import sample_correlated_gaussian
from estimators import estimate_mutual_information

dataset = torch.load('bit_string_dataset.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)


def compare_parities(timestep1, timestep2):
    parity1 = torch.sum(timestep1[:-1]) % 2
    parity2 = torch.sum(timestep2[:-1]) % 2
    return parity1 == parity2

def compare_extra_bit_parity(timestep1, timestep2):
    return timestep1[-1] == timestep2[-1]


count = 0
parity_count = 0
extra_parity_count = 0
one_count = 0
for batch in trainloader:
    for i in range(1, len(batch)):
        if compare_parities(batch[i][0], batch[i][1]):
            parity_count += 1
        if compare_extra_bit_parity(batch[i][0], batch[i][1]):
            extra_parity_count += 1
        if batch[i][0][-1] == 1:
            one_count += 1
        count += 1

print("Parity count: ", parity_count)
print("Extra parity count: ", extra_parity_count)
print("Total count: ", count)

print("Parity percentage: ", parity_count/count)
print("Extra parity percentage: ", extra_parity_count/count)


# %%


class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, input_dim1, input_dim2):
        super(SeparableCritic, self).__init__()
        self._g = nn.Sequential(
            nn.Linear(input_dim1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )
        self._h = nn.Sequential(
            nn.Linear(input_dim2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )
    

    def forward(self, x, y):
        
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores


# %%


def train_estimator(dataloader, **kwargs):
    critic = SeparableCritic(1,1)

    optimizer = optim.Adam(critic.parameters(), lr=1e-4)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project="smile_test", name=run_id)

    for batch in dataloader:
        x = batch[:,0]
        y = batch[:,1]

        last_digit_x = x[:, -1]
        last_digit_y = y[:, -1]

        # estimate mutual information
        mi = estimate_mutual_information('smile', last_digit_x, last_digit_y, critic, **kwargs)

        # compute loss
        loss = -mi

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({'MI': mi.item()})

    wandb.finish()

    return critic



train_estimator(trainloader, clip = 1)


# %%

data_params = {
    'dim': 1,
    'batch_size': 64,
    'cubic': None
}

critic_params = {
    'dim': 1,
    'layers': 2,
    'embed_dim': 32,
    'hidden_dim': 256,
    'activation': 'relu',
}

opt_params = {
    'iterations': 20000,
    'learning_rate': 5e-4,
}




x, y = sample_correlated_gaussian(
    dim=data_params['dim'], rho=0, batch_size=data_params['batch_size'], cubic=data_params['cubic'])

print(x.shape)
print(y.shape)

# %%

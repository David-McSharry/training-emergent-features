# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import wandb

from estimators import estimate_mutual_information


batch_size = 100

dataset = torch.load('bit_string_dataset_gp=0.99_ge=0.99_n=3000000.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def compare_parities(timestep1, timestep2):
    parity1 = torch.sum(timestep1[:-1]) % 2
    parity2 = torch.sum(timestep2[:-1]) % 2
    return parity1 == parity2

def compare_extra_bit_parity(timestep1, timestep2):
    return timestep1[-1] == timestep2[-1]


# count = 0
# parity_count = 0
# extra_parity_count = 0
# one_count = 0
# for batch in trainloader:
#     for i in range(1, len(batch)):
#         if compare_parities(batch[i][0], batch[i][1]):
#             parity_count += 1
#         if compare_extra_bit_parity(batch[i][0], batch[i][1]):
#             extra_parity_count += 1
#         if batch[i][0][-1] == 1:
#             one_count += 1
#         count += 1

# print("len dataset: ", count)

# print("Parity percentage: ", parity_count/count)
# print("Extra parity percentage: ", extra_parity_count/count)
# print("One percentage: ", one_count/count)


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
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores


print(SeparableCritic(1,1))

# %%

# import F
import torch.nn.functional as F

print(F.one_hot(torch.tensor([0,1,2,3,4,5])))
# %%


def train_estimator(dataloader, **kwargs):
    critic = SeparableCritic(1,7)

    optimizer = optim.Adam(critic.parameters(), lr=1e-4)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project="smile_test", name=run_id)

    for batch in dataloader:
        x = batch[:,0]
        y = batch[:,1].unsqueeze(2)

        batch_len = x.shape[0]

        one_hot_encoding = F.one_hot(torch.tensor([0,1,2,3,4,5])).unsqueeze(0).repeat(batch_len, 1, 1)
        y_with_one_hot = torch.cat((y, one_hot_encoding), dim=2)

        last_digit_x = x[:, -1].unsqueeze(1)
        last_digit_y = y_with_one_hot[:, -1]

        print(last_digit_x.shape)
        print(last_digit_y.shape)

        break
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



train_estimator(trainloader, clip = 10)


# %%

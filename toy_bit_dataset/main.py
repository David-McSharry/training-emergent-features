# %%
import torch
import matplotlib.pyplot as plt
from trainers import (train_model_A,
                        train_extra_bit_decoder,
                        train_parity_bit_decoder,
                        train_extra_bit_decoder,
                        train_parity_bit_decoder,
                        train_unsimilar_model_smile
                    )
import mutual_information as mi
import numpy as np


import lovely_tensors as lt

lt.monkey_patch()

batch_size = 1000

# dataset = torch.load('datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e7.pth')

dataset = torch.load('datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e7.pth')

# dataset0 = dataset[:-1]
# dataset1 = dataset[1:]
# # stack
# dataset = torch.stack((dataset0, dataset1), dim=1).float()

# X = dataset[:,0]
# Y = dataset[:,1]

# print(X[:5000, :2].shape)

# I = mi.pyMIestimator(X[:5000, :2], Y[:5000]) #default k = 5, base = np.exp(1)

# print('mutual information:', I)

# ch0 = X[:,0]

# I_0 = mi.pyMIestimator(ch0[:5000], Y[:5000]) #default k = 5, base = np.exp(1)   

# print('mutual information with first atom', I_0)

# ch2 = X[:,2]

# I_2 = mi.pyMIestimator(ch2[:5000], Y[:5000]) #default k = 5, base = np.exp(1)

# print('mutual information with third atom', I_2)

# # mutual info of third and first atom 

# I_2_0 = mi.pyMIestimator(ch2[:5000], ch0[:5000]) #default k = 5, base = np.exp(1)

# print('mutual information with third and first atom', I_2_0)

# # repeat the dataset 100 times
# dataset = dataset.repeat(100, 1, 1)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# %%

# train model A
model_A = train_model_A(trainloader)



# %%

from models import SupervenientFeatureNetwork
import numpy as np


model_A = SupervenientFeatureNetwork()
model_A.load_state_dict(torch.load('models/winneRRRRR.pt'))
model_A.eval()

model_B = train_unsimilar_model_smile(model_A, trainloader)




# %%

# %%

# save model
#
#
#
#

import time

# Get current time to use in file name
timestr = time.strftime("%Y%m%d-%H%M%S")

# Save model_B's state dict, appending the current time to the file name
torch.save(model_B.state_dict(), f'models/SUCCESS_MODEL_B_{timestr}.pt')

# %%




# compare vCLUB and SMILE MI estimates


from CLUB_estimator import CLUB, BinaryCLUB
import torch
import wandb
from SMILE_estimator import estimate_MI_smile
import torch.nn as nn
from datetime import datetime

class TestCritic1(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, x_dim, y_dim):
        super(TestCritic1, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(y_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, thing1, thing2):
        encoded1 = self.encoder1(thing1)
        encoded2 = self.encoder2(thing2)
        scores = torch.matmul(encoded1, encoded2.t())
        return scores

slices_of_bits = [
    # '[all bits]',
    # '[first 5 bits]',
    '[extra bit]'
]

batch_size = 1000

dataset = torch.load('datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e7.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def prepare_batch(batch, batch_size, slice_name):
    x0 = batch[:,0]
    x1 = batch[:,1]
    if slice_name == '[all bits]':
        x,y = x0, x1
        assert x.shape[1] == 6 and y.shape[1] == 6
        return x,y
    elif slice_name == '[first 5 bits]':
        x, y = x0[:,:5], x1[:,:5]
        assert x.shape[1] == 5 and y.shape[1] == 5
        return x,y
    elif slice_name == '[extra bit]':
        x, y = x0[:,-1].unsqueeze(1), x1[:,-1].unsqueeze(1)
        assert x.shape[1] == 1 and y.shape[1] == 1
        return x,y

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb.init(project="training-emergent-features", id=run_id)

for slice_name in slices_of_bits:

    if slice_name == '[all bits]':
        x_dim = 6
        y_dim = 6
    elif slice_name == '[first 5 bits]':
        x_dim = 5
        y_dim = 5
    elif slice_name == '[extra bit]':
        x_dim = 1
        y_dim = 1

    club = CLUB(x_dim, y_dim, 64)
    smile_critic = TestCritic1(x_dim, y_dim)
    binary_club = BinaryCLUB()
    binary_club_with_BCE_loss = BinaryCLUB()

    bce_loss = nn.BCELoss()

    club_optimizer = torch.optim.Adam(club.parameters(), lr=1e-4)
    smile_critic_optimizer = torch.optim.Adam(smile_critic.parameters(), lr=1e-4)
    binary_club_optimizer = torch.optim.Adam(binary_club.parameters(), lr=1e-4)
    binary_club_with_BCE_loss_optimizer = torch.optim.Adam(binary_club_with_BCE_loss.parameters(), lr=1e-4)


    batch_size = trainloader.batch_size

    i = 0
    for batch in trainloader:
        
        x_sample, y_sample = prepare_batch(batch, batch_size, slice_name)

        smile_critic.zero_grad()
        smile_scores = smile_critic(x_sample, y_sample)
        smile_MI = estimate_MI_smile(smile_scores)
        smile_loss = - smile_MI
        smile_loss.backward()
        smile_critic_optimizer.step()

        club.zero_grad()
        club_loss = club.learning_loss(x_sample, y_sample)
        club_loss.backward()
        club_optimizer.step()
        club_MI_estimation = club.forward(x_sample, y_sample)

        binary_club.zero_grad()
        binary_club_loss = binary_club.learning_loss(x_sample, y_sample)
        binary_club_loss.backward()
        binary_club_optimizer.step()
        binary_club_MI_estimation = binary_club.MI(x_sample, y_sample)

        binary_club_with_BCE_loss.zero_grad()
        y_pred = binary_club_with_BCE_loss.forward(x_sample)
        loss = bce_loss(y_pred, y_sample)
        loss.backward()
        binary_club_with_BCE_loss_optimizer.step()
        binary_club_with_BCE_loss_MI_estimation = binary_club_with_BCE_loss.MI(x_sample, y_sample)

        # print(f"{x_sample[0]} - {y_sample[0]} - {torch.exp(binary_club.learning_loss(x_sample[0], y_sample[0]))}")

        wandb.log({f"{slice_name} - vCLUB MI": club_MI_estimation.item()})
        wandb.log({f"{slice_name} - SMILE MI": smile_MI.item()})
        wandb.log({f"{slice_name} - Binary CLUB MI": binary_club_MI_estimation.item()})
        wandb.log({f"{slice_name} - Binary CLUB with BCE loss MI": binary_club_with_BCE_loss_MI_estimation.item()})

        wandb.log({f"{slice_name} - vCLUB loss": club_loss.item()})
        wandb.log({f"{slice_name} - Binary CLUB loss": binary_club_loss.item()})
        wandb.log({f"{slice_name} - Binary CLUB with BCE loss loss": loss.item()})
        wandb.log({f"{slice_name} - SMILE loss": smile_loss.item()})

        i += 1

        if i > 1000:
            break


# %%
        



for batch in trainloader:
    x_sample, y_sample = prepare_batch(batch, batch_size, '[extra bit]')
    print(club.get_mu_logvar(x_sample[2]))

    print(x_sample[2])
    break


        









# %%

# %%
        
# compares the representions of a model to a extra bit and also parity bit        
#
#
#
#



import matplotlib.pyplot as plt
import numpy as np

# f = SupervenientFeatureNetwork()
# f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_20240217-012416.pt'))

f = model_A

with torch.no_grad():
    for batch in trainloader:
        x0 = batch[:100,0]
        x1 = batch[:100,1]

        V1 = f(x1)

        for i in range(100):
            # round Ve


            print(f"{round(float(V1.squeeze()[i]),2)} - {np.sum(np.array(x1[i][:5])) % 2} - {x1[i][-1]}")
        # plot a histogram of the values in V1
        break
# %%


# Aanalyzing the reps learned by a B model
#
#
#
#
    
from models import SupervenientFeatureNetwork
import torch

f = SupervenientFeatureNetwork()
# load models/SUCCESS_MODEL_B._3pt

batch_size = 5000

dataset = torch.load('datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e7.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_20240217-012416.pt'))
points = []

for batch in trainloader:
    x0 = batch[:,0]
    x1 = batch[:,1]

    representations = f(x0)
    reps2 = f(x1)

    for i in range(5000):
        print(f"{int(x0[i][0])} {int(x0[i][1])} {int(x0[i][2])} {int(x0[i][3])} {int(x0[i][4])} {int(x0[i][5])} - {int(sum(x0[i][:5]) % 2)} - {round(float(representations.squeeze()[i]),2)} - {round(float(reps2.squeeze()[i]),2)}")
        points.append([round(float(representations.squeeze()[i]),2), round(float(reps2.squeeze()[i]),2)])

    break

# plot the points
import matplotlib.pyplot as plt
import numpy as np
points = np.array(points)
plt.scatter(points[:,0], points[:,1])
plt.show()

# count the number of times each point appears
from collections import Counter
counts = Counter(map(tuple, points))

print(counts)

# 2d histogram for the points and their counts

x = [key[0] for key in counts.keys()]
y = [key[1] for key in counts.keys()]
weights = list(counts.values())

# Create a 2D histogram
plt.hist2d(x, y, weights=weights, bins=[np.arange(0,0.15,0.01), np.arange(0,0.15,0.01)])
# log scale for heat


# Add a colorbar to the histogram
plt.colorbar(label='Count')

# Show the plot
plt.show()


# %%


# an experiment to find the MI between reps of SUCCESS_MODEL_B_20240217-012416 and EXTRA BIT because it seems like extra bit was somehow learned
#
#
#
#

import torch
from SMILE_estimator import estimate_MI_smile
from models import SupervenientFeatureNetwork, TestCritic
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import get_MI_between_f_representation_and_extra_bit

dataset = torch.load('datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e7.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=True)

f = SupervenientFeatureNetwork()
f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_20240217-012416.pt'))

get_MI_between_f_representation_and_extra_bit(f, trainloader)








# %%


# an experiment to find the MI between reps of SUCCESS_MODEL_B_2 and XOR BIT because it seems like extra bit was somehow learned
#
#
#
#

import torch
from SMILE_estimator import estimate_MI_smile
from models import SupervenientFeatureNetwork, TestCritic
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import get_MI_between_f_representation_and_XOR_bit

dataset = torch.load('datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e7.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=True)

f = SupervenientFeatureNetwork()
f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_3.pt'))


get_MI_between_f_representation_and_XOR_bit(f, trainloader)

# %%



extra_bit_decoder = train_extra_bit_decoder(trainloader, f)

parity_bit_decoder = train_parity_bit_decoder(trainloader, f)


# %%

# plot distro of representations of 6 digits whene their parity bit is 0 (blue) and 1 (orange)
#
#
#

# f = SupervenientFeatureNetwork()
# f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_3.pt'))


f = model_A

import matplotlib.pyplot as plt
import numpy as np

with torch.no_grad():
    for batch in trainloader:
        x0 = batch[:1000,0]
        x1 = batch[:1000,1]

        V1 = f(x1)

        extra_bits = x1[:,-1].unsqueeze(1)

        zero_array = []
        one_array = []


        for i in range(100):
            # round Ve


            print(f"{round(float(V1.squeeze()[i]),2)} - {extra_bits[i]}")

            if extra_bits[i] == 0:
                zero_array.append(V1[i])
            else:
                one_array.append(V1[i])

        # plot a histogram of the values in V1 making zero and one arrays different colors
        plt.hist(np.array(zero_array).flatten(), bins=20, label='parity_bit = zero', color='blue')
        plt.hist(np.array(one_array).flatten(), bins=20, label='parity_bit = one', color='orange')
        # print the height of the bins
        plt.ylabel('Frequency')
        plt.xlabel('f(extra bit)')    
        plt.legend()
        plt.show()
        break


# %%


# Find MI between the reps and parity bit
#
#
from models import TestCritic, SupervenientFeatureNetwork
import torch

f = SupervenientFeatureNetwork()
f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_20240217-012416.pt'))

from utils import get_MI_between_f_representation_and_parity_bit

get_MI_between_f_representation_and_parity_bit(f, trainloader)



# %%

        


# f = SupervenientFeatureNetwork()
# f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_3.pt'))



f = model_A



with torch.no_grad():
    for batch in trainloader:
        x0 = batch[:100,0]
        x1 = batch[:100,1]

        V1 = f(x1)

        parity_batch = torch.sum(x1[:,:5], dim=1) % 2

        zero_array = []
        one_array = []


        for i in range(100):
            # round Ve


            print(f"{round(float(V1.squeeze()[i]),2)} - {parity_batch[i]}")

            if parity_batch[i] == 0:
                zero_array.append(V1[i])
            else:
                one_array.append(V1[i])

        
                
        zero_array = np.round(np.array(zero_array),2)
        one_array = np.round(np.array(one_array),2)

        print(zero_array.squeeze())
        print(one_array.squeeze())

        # plot a histogram of the values in V1 making zero and one arrays different colors
        plt.hist(zero_array.flatten(), bins=20, label='extra_bit = zero', color='blue')
        plt.hist(one_array.flatten(), bins=20, label='extra_bit = one', color='orange')
        # print the height of the bins
        plt.ylabel('Frequency')
        plt.xlabel('f(parity bit)')
        plt.legend()
        plt.show()

        break
    

# %%





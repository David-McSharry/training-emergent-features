# %%
import torch

from trainers import (train_model,
                        train_extra_bit_decoder,
                        train_parity_bit_decoder,
                        train_extra_bit_decoder,
                        train_parity_bit_decoder,
                        train_unsimilar_model
                    )

batch_size = 500

dataset = torch.load('datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e7.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# %%


f = train_model(trainloader)




# %%

from models import SupervenientFeatureNetwork
import numpy as np


model_A = SupervenientFeatureNetwork()
model_A.load_state_dict(torch.load('models/winneRRRRR.pt'))
model_A.eval()


model_B = train_unsimilar_model(model_A, trainloader)


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


# Compare encodings of a 6 digit set between two different models
#
#
#
#

from models import SupervenientFeatureNetwork
import numpy as np

f1 = SupervenientFeatureNetwork()
f1.load_state_dict(torch.load('models/winneRRRRR.pt'))

f2 = SupervenientFeatureNetwork()
f2.load_state_dict(torch.load('models/learned_parity_bit_f_VMI_only.pt'))

for batch in trainloader:
    x0 = batch[:100,0]
    x1 = batch[:100,1]

    V1 = f1(x1)
    V2 = f2(x1)

    for i in range(100):
        print(f"{round(float(V1.squeeze()[i]),2)} - {round(float(V2.squeeze()[i]),2)}")
    




# %%
        
# compares the representions of a model to a extra bit and also parity bit        
#
#
#
#



import matplotlib.pyplot as plt
import numpy as np

f = SupervenientFeatureNetwork()
f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_3.pt'))

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


f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_3.pt'))
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


# an experiment to find the MI between reps of SUCCESS_MODEL_B_2 and EXTRA BIT because it seems like extra bit was somehow learned
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
f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_3.pt'))

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

f = SupervenientFeatureNetwork()
f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_3.pt'))

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
f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_3.pt'))

from utils import get_MI_between_f_representation_and_parity_bit

get_MI_between_f_representation_and_parity_bit(f, trainloader)



# %%

        


f = SupervenientFeatureNetwork()
f.load_state_dict(torch.load('models/SUCCESS_MODEL_B_3.pt'))



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




# %%
import torch

from einops import rearrange, reduce, repeat

from trainers import (train_model_A,
                        train_extra_bit_decoder,
                        train_parity_bit_decoder,
                        train_extra_bit_decoder,
                        train_parity_bit_decoder,
                        train_unsimilar_model_smile
                    )

import torch.nn as nn
import torch.optim as optim

batch_size = 1000


dataset = torch.load('datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e7.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



# %%
# ----------------------------------------


from CLUB_estimator import BinaryCLUB
import wandb
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb.init(project="training-emergent-features", id=run_id)


vclub0 = BinaryCLUB()
bce_loss = nn.BCELoss()
vclub1 = BinaryCLUB()



optim_vclub0 = torch.optim.Adam(vclub0.parameters(), lr=1e-4)
optim_vclub1 = torch.optim.Adam(vclub1.parameters(), lr=1e-4)

i = 0    

for batch in trainloader:


    x0 = batch[:, 0]
    x1 = batch[:, 1]


    digit_0 = x0[:, -1].unsqueeze(1) # (batch_size, 1)
    digit_1 = x1[:, -1].unsqueeze(1) # ^


    vclub0.zero_grad()
    dig1_pred = vclub0.forward(digit_0)
    loss0 = bce_loss(dig1_pred, digit_1)
    loss0.backward()
    optim_vclub0.step()


    vclub1.zero_grad()
    loss1 = vclub0.learning_loss(digit_0, digit_1)
    loss1.backward()
    optim_vclub1.step()

    # print('vclub', float(digit_1[0]), float(vclub1.forward(digit_0)[0]))
    # print('vclub with BCE', float(digit_1[0]), float(vclub0.forward(digit_0)[0]))


    wandb.log({"BCE_loss": loss0.item()})
    wandb.log({"regular_loss": loss1.item()})

    wandb.log({"BCE MI": vclub0.MI(digit_0, digit_1).item()})
    wandb.log({"regular MI": vclub1.MI(digit_0, digit_1).item()})


    accuracy = (vclub0.forward(digit_0) > 0.5).float().eq(digit_1).float().mean()
    print(f"Accuracy 0: {accuracy.item()}")

    accuracy = (vclub1.forward(digit_0) > 0.5).float().eq(digit_1).float().mean()
    print(f"Accuracy 1: {accuracy.item()}")

    i += 1

    if i > 1000:
        break





# %%

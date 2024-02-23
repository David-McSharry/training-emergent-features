import matplotlib.pyplot as plt
import numpy as np
from models import TestCritic
import torch
import torch.optim as optim
from SMILE_estimator import estimate_MI_smile


def get_MI_between_f_representation_and_parity_bit(f, trainloader):

    critic = TestCritic()
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

    for batch in trainloader:

        x1_batch = batch[:, 1]

        assert x1_batch.size()[1] == 6

        parity_bit_batch = (torch.sum(x1_batch[:, :5], dim = 1) % 2).unsqueeze(1)

        scores = critic(parity_bit_batch, f(x1_batch))

        MI = estimate_MI_smile(scores)
        loss = -MI

        critic_opt.zero_grad()
        loss.backward()
        critic_opt.step()

        print(MI)

def get_MI_between_f_representation_and_extra_bit(f, trainloader):
    
    critic = TestCritic()
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

    
    for batch in trainloader:

        x1 = batch[:,1]


        x1_rep = f(x1)

        scores = critic(x1[:,5].unsqueeze(1), x1_rep)

        MI = estimate_MI_smile(scores)
        
        loss = -MI

        critic_opt.zero_grad()
        loss.backward()
        critic_opt.step()

        print(MI)


def get_MI_between_f_representation_and_XOR_bit(f, trainloader):
    
    critic = TestCritic()
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

    
    for batch in trainloader:

        x1_batch = batch[:,1]
        
        parity_bit_batch = (torch.sum(x1_batch[:, :5], dim = 1) % 2).unsqueeze(1)
        extra_bit_batch = x1_batch[:,5].unsqueeze(1)

        XOR_parity_bit_batch = (parity_bit_batch + extra_bit_batch) % 2

        x1_rep = f(x1_batch)

        scores = critic(XOR_parity_bit_batch, x1_rep)


        MI = estimate_MI_smile(scores)
        
        loss = -MI

        critic_opt.zero_grad()
        loss.backward()
        critic_opt.step()

        print(MI)

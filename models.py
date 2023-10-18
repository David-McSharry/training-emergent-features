
import torch
from torch import nn

class SupervenientFeatureNetwork(nn.Module):
    def __init__(self):
        super(SupervenientFeatureNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class CriticNetworkDecoupled(nn.Module):
    def __init__(self):
        super(CriticNetworkDecoupled, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 1) # output layer

    def forward(self, v1, v2):
        # Concatenate v and x along the second dimension
        z = torch.cat([v1, v2], dim=1)
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = self.fc4(z) # no activation on the output layer
        return z

class CriticNetworkDownward(nn.Module):
    def __init__(self):
        super(CriticNetworkDownward, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 1) # output layer

    def forward(self, v, x):
        z = torch.cat([v, x.unsqueeze(1)], dim=1)
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = self.fc4(z) # no activation on the output layer
        return z

class MainNetwork(nn.Module):
    def __init__(self):
        super(MainNetwork, self).__init__()
        self.f_supervenient = SupervenientFeatureNetwork()
        self.causal_decoupling_critic = CriticNetworkDecoupled()
        self.downward_causation_critic = CriticNetworkDownward()



import torch
import torch.nn as nn

class SupervenientFeatureNetwork(nn.Module):
    def __init__(self):
        super(SupervenientFeatureNetwork, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.f(x)

    # think harder about this if it is needed    
    # def get_feature_Hilbert_rep(self, dataloader):
    #     """Get the feature Hilbert representation of the dataset and return it as one tensor"""
    #     output_tensors = []
    #     for batch in dataloader:
    #         x0 = batch[:,0]
    #         output_tensors.append(self.f(x0))
    #     return torch.cat(output_tensors, dim=0)
    
    
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


class MainNetwork(nn.Module):
    def __init__(self):
        super(MainNetwork, self).__init__()
        self.supervenient_feature_network = SupervenientFeatureNetwork()
        self.critic_g = SeparableCritic(1, 1)
        self.critic_h = SeparableCritic(1, 7)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

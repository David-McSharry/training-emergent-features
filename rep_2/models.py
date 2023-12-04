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
        
    
class PredSeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self):
        super(PredSeparableCritic, self).__init__()
        self.v_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

        self.W = nn.Linear(8, 8, bias=False)

    def forward(self, v0, v1):
        v0_encoded = self.v_encoder(v0)
        v1_encoded = self.v_encoder(v1)
        v1_encoded_transformed = self.W(v1_encoded)

        scores = torch.matmul(v0_encoded, v1_encoded_transformed.t())
        return scores
    

class MarginalSeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self):
        super(MarginalSeparableCritic, self).__init__()
        self.v_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

        self.xi_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

        self.W = nn.Linear(8, 8, bias=False)

    def forward(self, x0i, v1):
        v1_encoded = self.v_encoder(v1)
        x0i_encoded = self.xi_encoder(x0i)
        v1_encoded_transformed = self.W(v1_encoded)

        scores = torch.matmul(x0i_encoded, v1_encoded_transformed.t())

        return scores
        

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

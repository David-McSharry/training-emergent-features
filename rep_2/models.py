import torch
import torch.nn as nn


class TestCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self):
        super(TestCritic, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(1, 128),
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

    def forward(self, x0i, v1):
        v1_encoded = self.v_encoder(v1)
        x0i_encoded = self.xi_encoder(x0i)
        scores = torch.matmul(x0i_encoded, v1_encoded.t())

        return scores

class MarginalSeperableCriticExpanded(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self):
        super(MarginalSeperableCriticExpanded, self).__init__()
        self.v_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 264),
            nn.ReLU(),
            nn.Linear(264, 264),
            nn.ReLU(),
            nn.Linear(264, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

        self.xi_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 264),
            nn.ReLU(),
            nn.Linear(264, 264),
            nn.ReLU(),
            nn.Linear(264, 128),
            nn.ReLU(),
            nn.Linear(128, 16),

        )

    def forward(self, x0i, v1):
        v1_encoded = self.v_encoder(v1)
        x0i_encoded = self.xi_encoder(x0i)
        scores = torch.matmul(x0i_encoded, v1_encoded.t())

        return scores
    

class DifferentRepCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self):
        super(DifferentRepCritic, self).__init__()
        self.v_A_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

        self.v_B_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, v0, v1):
        v0_encoded = self.v_A_encoder(v0)
        v1_encoded = self.v_B_encoder(v1)

        scores = torch.matmul(v0_encoded, v1_encoded.t())
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

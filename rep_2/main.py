# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb


from utils import *
from estimators import estimate_mutual_information

# define the dimension of the Gaussian

dim = 1

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
    
class ConcatCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, dim, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        # input dim is 2 * dim, then hidden layer 128 dim, then hidden layer 64 dim, then hidden layer 8 dim, then output layer 1 dim
        self._f = nn.Sequential(
            nn.Linear(2 * dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x, y):
        batch_size = x.size(0)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()




# %%

def train_estimator(critic_params, data_params, mi_params, opt_params, **kwargs):
    """Main training loop that estimates time-varying MI."""

    critic_h = ConcatCritic(1).to(device='cpu')
    critic_g = ConcatCritic(1).to(device='cpu')

    supervenient_feature_network = SupervenientFeatureNetwork().to(device='cpu')

    wandb.init(project="training-emergent-features", id="test")

    opt_crit_h = optim.Adam(critic_h.parameters(), lr=opt_params['learning_rate'])
    opt_crit_g = optim.Adam(critic_g.parameters(), lr=opt_params['learning_rate'])
    opt_supervenient_feature_network = optim.Adam(supervenient_feature_network.parameters(), lr=opt_params['learning_rate'])

    def train_step(X0, X1, **kwargs):
        opt_crit_h.zero_grad()
        opt_crit_g.zero_grad()
        opt_supervenient_feature_network.zero_grad()

        V0 = supervenient_feature_network(X0)
        V1 = supervenient_feature_network(X1)

        causal_decoupling_MI = estimate_mutual_information('smile', V0, V1, critic_h, **kwargs)
        
        downward_causation_MI = 0
        for i in range(6):
            downward_causation_MI += estimate_mutual_information('smile', V1, X0[:, i], critic_g, **kwargs)

        Psi_loss = - (causal_decoupling_MI - downward_causation_MI)

        # normalize the weights
        # l2_lambda = 0.01
        # l2_reg = torch.tensor(0.)
        # for param in critic_h.parameters():
        #     l2_reg += torch.norm(param)
        # for param in critic_g.parameters():
        #     l2_reg += torch.norm(param)
        # for param in supervenient_feature_network.parameters():
        #     l2_reg += torch.norm(param)
        # Psi_loss += l2_lambda * l2_reg


        wandb.log({"Psi loss": Psi_loss})

        Psi_loss.backward()
        opt_crit_h.step()
        opt_crit_g.step()
        opt_supervenient_feature_network.step()

        return Psi_loss


    dataset = torch.load('data/bit_string_dataset.pth')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    estimates = []
    for batch in trainloader:
        mi = train_step(batch[:,0], batch[:,1], **kwargs)
        mi = mi.detach().cpu().numpy()
        estimates.append(mi)

    wandb.finish()

    return np.array(estimates)


# %%
data_params = {
    'dim': 1,
    'batch_size': 100, # not used
    'cubic': None  # not used
}

critic_params = {
    'dim': 1,
    'layers': 1, # not used
    'embed_dim': 32, # not used
    'hidden_dim': 256, # not used
    'activation': 'relu', # not used
}

opt_params = {
    'iterations': 20000, # not used
    'learning_rate': 1e-4,
}



# %%

mi_numpys = dict()

for critic_type in ['concat']:
    mi_numpys[critic_type] = dict()

    estimator = 'smile'
    for i, clip in enumerate([999]):
        mi_params = dict(estimator=estimator, critic=critic_type, baseline='unnormalized')
        mis = train_estimator(critic_params, data_params, mi_params, opt_params, clip=clip)
        mi_numpys[critic_type][f'{estimator}_{clip}'] = mis


# %%

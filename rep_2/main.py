# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datetime import datetime
from tqdm import tqdm


from estimators import estimate_mutual_information

batch_size = 100

dataset = torch.load('bit_string_dataset_gp=0.99_ge=0.99_n=3000000.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



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
        

# %%


def train_model(dataloader, **kwargs):
    """Main training loop that estimates time-varying MI."""

    model = MainNetwork().to('cpu')

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project="training-emergent-features", id=run_id)


    # track gradients in our three models to make sure they are being updated
    wandb.watch(model.critic_h)
    wandb.watch(model.critic_g)
    wandb.watch(model.supervenient_feature_network)

    opt_crit_h = optim.Adam(model.critic_h.parameters(), lr=1e-4)
    opt_crit_g = optim.Adam(model.critic_g.parameters(), lr=1e-4)
    opt_supervenient_feature_network = optim.Adam(model.supervenient_feature_network.parameters(), lr=1e-4)

    for batch in dataloader:
        x1 = batch[:,0]
        x0 = batch[:,1]

        batch_len = x1.shape[0]
        
        one_hot_encoding = F.one_hot(torch.tensor([0,1,2,3,4,5])).unsqueeze(0).repeat(batch_len, 1, 1)
        x0_with_one_hot = torch.cat((x0.unsqueeze(2), one_hot_encoding), dim=2)
        
        V0 = model.supervenient_feature_network(x1)
        V1 = model.supervenient_feature_network(x0)

        digit_0_x0_with_one_hot = x0_with_one_hot[:, 0]
        digit_1_x0_with_one_hot = x0_with_one_hot[:, 1]
        digit_2_x0_with_one_hot = x0_with_one_hot[:, 2]
        digit_3_x0_with_one_hot = x0_with_one_hot[:, 3]
        digit_4_x0_with_one_hot = x0_with_one_hot[:, 4]
        last_digit_x0_with_one_hot = x0_with_one_hot[:, 5]

        psi = 0

        psi += estimate_mutual_information('smile', V0, V1, model.critic_g, **kwargs)

        eps = 1e-5
        # psi -= eps * estimate_mutual_information('smile', V1, digit_0_x0_with_one_hot, model.critic_h, **kwargs)
        # psi -= eps * estimate_mutual_information('smile', V1, digit_1_x0_with_one_hot, model.critic_h, **kwargs)
        # psi -= eps * estimate_mutual_information('smile', V1, digit_2_x0_with_one_hot, model.critic_h, **kwargs)
        # psi -= eps * estimate_mutual_information('smile', V1, digit_3_x0_with_one_hot, model.critic_h, **kwargs)
        # psi -= eps * estimate_mutual_information('smile', V1, digit_4_x0_with_one_hot, model.critic_h, **kwargs)
        # psi -= eps * estimate_mutual_information('smile', V1, last_digit_x0_with_one_hot, model.critic_h, **kwargs)

        # compute loss
        loss = psi

        # backprop
        opt_crit_h.zero_grad()
        opt_crit_g.zero_grad()
        opt_supervenient_feature_network.zero_grad()
        loss.backward()

        opt_crit_h.step()
        opt_crit_g.step()
        opt_supervenient_feature_network.step()

        wandb.log({'Psi': psi.item()})
    
    wandb.finish()

    return model


model = train_model(trainloader, clip = 10)




# %%


f = model.supervenient_feature_network
with torch.no_grad():
    for batch in trainloader:
        x0 = batch[:15,0]
        x1 = batch[:15,1]

        V0 = f(x0)
        V1 = f(x1)

        print(V0)
        print(V1)
        print(x0)
        print(x1)
        break


# %%

mi_numpys = dict()

for critic_type in ['concat']:
    mi_numpys[critic_type] = dict()

    estimator = 'smile'
    for i, clip in enumerate([999]):
        mi_params = dict(estimator=estimator, critic=critic_type, baseline='unnormalized')
        mis = train_model(critic_params, data_params, mi_params, opt_params, clip=clip)
        mi_numpys[critic_type][f'{estimator}_{clip}'] = mis


# %%

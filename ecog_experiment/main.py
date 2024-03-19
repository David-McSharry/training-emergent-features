
# %%
import torch
import matplotlib.pyplot as plt
from models import SupervenientEncoder, DownwardSmile, DecoupledSmile
import wandb
from datetime import datetime
import torch.optim as optim
from utils import prepare_batch


config = {
    'dataset_path': 'data/ecog_dataset.pth',
    'batch_size': 100,
    'num_features' : 6,
    'V_dim' : 1,
    'f_config': {
        'lr': 1e-4
    },
    'downward_config': {
        'critic_output_dim': 8,
        'lr': 1e-4
    },
    'decoupled_config': {
        'critic_output_dim': 8,
        'lr': 1e-5
    }
}

dataset = torch.load(config['dataset_path'])
trainloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

f = SupervenientEncoder(
    num_features=config['num_features'],
    V_dim=config['V_dim']
)

downward = DownwardSmile(
    V_dim=config['V_dim'],
    critic_ouput_dim=config['downward_config']['critic_output_dim'],
    feature_num=config['num_features']
)

decoupled = DecoupledSmile(
    V_dim=config['V_dim'],
    critic_ouput_dim=config['decoupled_config']['critic_output_dim'],
)

opt_f = optim.Adam(f.parameters(), lr = config['f_config']['lr'])
opt_downward = optim.Adam(downward.parameters(), lr = config['downward_config']['lr'])
opt_decoupled = optim.Adam(decoupled.parameters(), lr = config['decoupled_config']['lr'])

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb.init(project="training-emergent-features", id=run_id)
wandb.watch(downward)
wandb.watch(decoupled)
wandb.watch(f)

for batch in trainloader:
    batch = prepare_batch(batch)
    X0 = batch[:, 0].float()
    X1 = batch[:, 1].float()

    V0 = f(X0)
    V1 = f(X1)

    opt_downward.zero_grad()
    opt_decoupled.zero_grad()
    opt_f.zero_grad()
    decoupled_loss = decoupled.learning_loss(V0, V1)
    decoupled_loss.backward(retain_graph=True)
    opt_decoupled.step()

    opt_downward.zero_grad()
    opt_decoupled.zero_grad()
    opt_f.zero_grad()
    downward_loss = downward.learning_loss(X0, V1)
    downward_loss.backward(retain_graph=True)
    opt_downward.step()

    opt_downward.zero_grad()
    opt_decoupled.zero_grad()
    opt_f.zero_grad()
    f_loss = f.learning_loss(X0, X1, downward, decoupled)
    f_loss.backward(retain_graph=True)
    opt_f.step()

    wandb.log({'f_loss': f_loss.item()})
    wandb.log({'downward_loss': downward_loss.item()})
    wandb.log({'decoupled_loss': decoupled_loss.item()})
    wandb.log({'decoupled MI': decoupled.get_MI(V0, V1).item()})
    wandb.log({'downward MI': downward.get_sum_MI(X0, V1).item()})





# %%
import torch.nn as nn

confoig = {
  "input_size": 10,
  "hidden_size": 50,
  "output_size": 1,
  "num_layers": 2,
  "learning_rate": 0.001,
  "num_epochs": 50,
  "sequence_length": 100,
  "batch_size": 64
}



class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

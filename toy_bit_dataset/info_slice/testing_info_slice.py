import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import utils_data, utils_os
import torch.nn.functional as F

import mi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def g1(y):    # linear mapping
    return y
    
def g2(y):    # nonlinear mapping, where z = [e(Ay), tanh(Ay)] 
    A = torch.Tensor([[1, 1.5], [1.5, 1]]).to(y.device)
    q = torch.matmul(y, A)
    z = torch.zeros(y.size()).to(y.device)
    z[:, 0] = q[:, 0].exp()
    z[:, 1] = torch.tanh(q[:, 1])
    return z

g = g1


# hyperparams
class Hyperparams(utils_os.ConfigDict):
    def __init__(self): 
        self.lr = 1e-3
        self.bs = 250
        self.n_slice = 50               # <-- we can tune this to adjust performance
hyperparams=Hyperparams()

data_dim = 1


mi_estimator = mi.SliceInfominLayer([data_dim, hyperparams.n_slice, data_dim], hyperparams=hyperparams).to(device)

batch_size = 1000

dataset = torch.load('/Users/davidmcsharry/dev/imperial/training-emergent-features/ecog_experiment/data/ecog_dataset.pth')
dataset0 = dataset[:-1]
dataset1 = dataset[1:]
# stack
dataset = torch.stack((dataset0, dataset1), dim=1).float()

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# dataset
# sampler = GaussianSampler(sample_dim).cuda()
# sampler_optimizer = torch.optim.Adam(sampler.parameters(), lr = hyperparams.lr)

# # estimator for I(x, y)
mi_estimator = mi.SliceInfominLayer([6, hyperparams.n_slice, 6], hyperparams=hyperparams).to(device)

def estimate_MI_smile(scores):
    """
    Returns the MI estimate using the SMILE estimator given the scores matrix and a clip
    """
    clip = 5
    
    first_term = scores.diag().mean()

    batch_size = scores.size(0)

    # clip scores between -clip and clip
    clipped_scores = torch.clamp(scores, -clip, clip)

    # e^clipped_scores
    exp_clipped_scores = torch.exp(clipped_scores)

    mask = (torch.ones_like(exp_clipped_scores) - torch.eye(batch_size)).to(device=exp_clipped_scores.device)

    masked_exp_clipped_scores = exp_clipped_scores * mask

    num_non_diag = mask.sum()

    mean_exp_clipped_scores = masked_exp_clipped_scores.sum() / num_non_diag

    second_term = torch.log2(mean_exp_clipped_scores)

    return (1/torch.log(torch.tensor(2.0))) * first_term - second_term


def get_causally_decoupled_MI(pred_critic, x0, x1, f):
    """
    A loss function that returns the MI between a supervenient feature
    V1 = f(x1) and the supervenient feature V0 = f(x0)
    """
    V1 = f(x1)
    V0 = f(x0)
    scores = pred_critic(V0, V1)
    V_mutual_info = estimate_MI_smile(scores)
    return V_mutual_info


for batch in trainloader:

    x0 = batch[:,0]
    x1 = batch[:,1]

    # batch_len = x1.shape[0]

    # one_hot_encoding = F.one_hot(torch.tensor([0,1,2,3,4,5])).unsqueeze(0).repeat(batch_len, 1, 1)
    # x0_with_one_hot = torch.cat((x0.unsqueeze(2), one_hot_encoding), dim=2)

    # x0_extra_bit_with_one_hot = x0_with_one_hot[:,-1,:]

    # x1_exta_bit = x1[:,-1].unsqueeze(1)
    # x0_extra_bit = x0[:,-1].unsqueeze(1)

    # print(x0_extra_bit_with_one_hot.shape)

    # print(x1_exta_bit.shape)
    print(mi_estimator.learn(x0, x1))

    break






# def train_model_A_with_info_slice(dataloader):
#     pred_critic = PredSeparableCritic()
#     f = SupervenientFeatureNetwork()

#     print("Pred Critic")
#     print(pred_critic)
#     print("Supervenient Feature Network")
#     print(f)

#     run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

#     wandb.init(project="training-emergent-features", id=run_id)
    

#     opt_pred = optim.Adam(pred_critic.parameters(), lr=1e-4)
#     opt_f = optim.Adam(f.parameters(), lr=1e-5)


#     for batch in tqdm(dataloader):

#         x0 = batch[:,0]
#         x1 = batch[:,1]

#         batch_len = x1.shape[0]

#         one_hot_encoding = F.one_hot(t.tensor([0,1,2,3,4,5])).unsqueeze(0).repeat(batch_len, 1, 1)
#         x0_with_one_hot = torch.cat((x0.unsqueeze(2), one_hot_encoding), dim=2)



#         f.zero_grad()
#         pred_critic.zero_grad()


#         V_mutual_info = get_causally_decoupled_MI(pred_critic, x0, x1, f)
#         g_loss = - V_mutual_info

#         opt_pred.zero_grad()
#         g_loss.backward(retain_graph=True)
#         opt_pred.step()


#         # marginal terms







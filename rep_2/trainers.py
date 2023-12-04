import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datetime import datetime
from models import SupervenientFeatureNetwork, PredSeparableCritic, MarginalSeparableCritic
from models import Decoder
import torch.nn as nn
from tqdm import tqdm
from rep_2.SMILE_estimator import estimate_MI_smile

def get_sum_margininal_MIs(marginal_critic, x0_one_hot, x1, f, config):
    """
    A loss function that returns the sum of the MIs between a supervenient feature
    V1 = f(x1) and each digit of the 6 binary digits
    """
    sum_margininal_MIs = 0
    V1 = f(x1)
    for i in range(6):
        scores = marginal_critic(x0_one_hot[:,i], V1)
        sum_margininal_MIs += estimate_MI_smile(scores, config['clip'])
    return sum_margininal_MIs


def get_V_mutual_info(pred_critic, x0, x1, f, config):
    """
    A loss function that returns the MI between a supervenient feature
    V1 = f(x1) and the supervenient feature V0 = f(x0)
    """
    V1 = f(x1)
    V0 = f(x0)
    scores = pred_critic(V0, V1)
    V_mutual_info = estimate_MI_smile(scores, config['clip'])
    return V_mutual_info


def train_model(dataloader):

    pred_critic = PredSeparableCritic()
    marginal_critic = MarginalSeparableCritic()
    f = SupervenientFeatureNetwork()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project="training-emergent-features", id=run_id)

    # track gradients in our three models to make sure they are being updated
    wandb.watch(pred_critic)
    wandb.watch(marginal_critic)
    wandb.watch(f)

    # init optimizer
    opt_pred = optim.Adam(pred_critic.parameters(), lr=1e-4)
    opt_marginal = optim.Adam(marginal_critic.parameters(), lr=1e-4)
    opt_f = optim.Adam(f.parameters(), lr=1e-5)
    
    for batch in tqdm(dataloader):
        x0 = batch[:,0]
        x1 = batch[:,1]

        batch_len = x1.shape[0]

        one_hot_encoding = F.one_hot(torch.tensor([0,1,2,3,4,5])).unsqueeze(0).repeat(batch_len, 1, 1)
        x0_with_one_hot = torch.cat((x0.unsqueeze(2), one_hot_encoding), dim=2) # (batch_size, 6, 7)

        f.zero_grad()
        pred_critic.zero_grad()
        marginal_critic.zero_grad()


        # pred terms

        V_mutual_info = get_V_mutual_info(pred_critic, x0, x1, f)
        g_loss = - V_mutual_info

        opt_pred.zero_grad()
        g_loss.backward(retain_graph=True)
        opt_pred.step()


        # marginal terms

        sum_margininal_MIs = get_sum_margininal_MIs(marginal_critic, x0_with_one_hot, x1, f)
        marginal_loss = - sum_margininal_MIs

        opt_marginal.zero_grad()
        marginal_loss.backward(retain_graph=True)
        opt_marginal.step()


        # f

        psi = get_V_mutual_info(pred_critic, x0, x1, f) - get_sum_margininal_MIs(marginal_critic, x0_with_one_hot, x1, f)
        f_loss = - psi

        opt_f.zero_grad()
        f_loss.backward()
        opt_f.step()


        wandb.log({'Psi': psi.item()})
        wandb.log({'Sum of Marginal MIs': sum_margininal_MIs.item()})
        wandb.log({'V Mutual Info': V_mutual_info.item()})

    wandb.finish()

    return f


def CKA_function(X, Y):
    assert X.shape[0] == Y.shape[0]
    X = X - torch.mean(X, dim=0)
    Y = Y - torch.mean(Y, dim=0)

    XTX = X.t() @ X
    YTY = Y.t() @ Y
    XTY = X.t() @ Y

    result = XTY.norm(p="fro") ** 2 / (XTX.norm(p="fro") * YTY.norm(p="fro"))

    del XTX, YTY, XTY
    torch.cuda.empty_cache()

    return result


# def train_unsimilar_model(
#     model_A, trainloader, CKA_weight, track_CKA=False, device='cpu', **kwargs
# ):
# TODO: implement


def train_extra_bit_decoder(dataloader, f_network):
    """Main training loop that estimates time-varying MI."""

    num_steps = 10000

    model = Decoder().to('cpu')

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project="training-emergent-features", id=run_id)


    # track gradients in our three models to make sure they are being updated
    wandb.watch(model)

    opt = optim.Adam(model.parameters(), lr=1e-4)

    counter = 0
    for batch in dataloader:
        counter += 1
        if counter > num_steps:
            break
        x1 = batch[:,1]

        V1 = f_network(x1)

        loss = nn.MSELoss()

        # compute loss
        loss = loss(model(V1), x1[:,-1].unsqueeze(1))

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        wandb.log({'Loss': loss.item()})

    wandb.finish()

    return model


def train_parity_bit_decoder(dataloader, f_network):
    """Main training loop that estimates time-varying MI."""

    num_steps = 10000

    model = Decoder().to('cpu')

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project="training-emergent-features", id=run_id)


    # track gradients in our three models to make sure they are being updated
    wandb.watch(model)

    opt = optim.Adam(model.parameters(), lr=1e-4)

    counter = 0
    for batch in dataloader:
        counter += 1
        if counter > num_steps:
            break

        x1 = batch[:,1]

        parity_batch = torch.sum(x1[:,:5], dim=1) % 2

        V1 = f_network(x1)

        loss = nn.MSELoss()

        # compute loss
        loss = loss(model(V1), parity_batch.unsqueeze(1))

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        wandb.log({'Loss': loss.item()})



    wandb.finish()

    return model

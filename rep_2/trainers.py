import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datetime import datetime
from models import SupervenientFeatureNetwork, PredSeparableCritic, MarginalSeparableCritic
from models import Decoder
import torch.nn as nn
from tqdm import tqdm
from custom_estimators import estimate_MI_smile

def get_sum_margininal_MIs(marginal_critic, x0_one_hot, x1, f):
    """
    A loss function that returns the sum of the MIs between a supervenient feature
    V1 = f(x1) and each digit of the 6 binary digits
    """
    sum_margininal_MIs = 0
    V1 = f(x1)
    for i in range(6):
        scores = marginal_critic(x0_one_hot[:,i], V1)
        sum_margininal_MIs += estimate_MI_smile(scores)
    return sum_margininal_MIs


def get_V_mutual_info(pred_critic, x0, x1, f):
    """
    A loss function that returns the MI between a supervenient feature
    V1 = f(x1) and the supervenient feature V0 = f(x0)
    """
    V1 = f(x1)
    V0 = f(x0)
    scores = pred_critic(V0, V1)
    V_mutual_info = estimate_MI_smile(scores)
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
        # wandb.log({'extra bit MI': get_marginal_MI_last_bit(critic_h, x0_with_one_hot, x1, supervenient_feature_network, **kwargs)})

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


def train_unsimilar_model(
    model_A, trainloader, CKA_weight, track_CKA=False, device='cpu', **kwargs
):
    """
    A model that takes in a model A, and trains a new model B
    that is dissilimar to model A by the CKA metric. The CKA for the
    full Hilbert space rep is not calculated as taking the rep over the 
    batch and making minimizing the CKA between those is enough to ensure the full 
    CKA is small.

    The CKA loss is added onto the original loss so you get a model that 
    optomizes for the original loss and the is dissimilar to model A.

    args:
        base_model: model A
        trainloader: binary digit dataset
        track_CKA: whether to track CKA between model A and model B
        device: device to train on
        **kwargs: pretty much just clip value
    """
    model = MainNetwork()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    supervenient_A_network = model_A.supervenient_feature_network

    run_id = "ModelB_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project="training-emergent-features", id=run_id)
    wandb.watch(model)

    for batch in trainloader:
        x0 = batch[:,0]
        x1 = batch[:,1]

        batch_len = x1.shape[0]
        
        one_hot_encoding = F.one_hot(torch.tensor([0,1,2,3,4,5])).unsqueeze(0).repeat(batch_len, 1, 1)
        x0_with_one_hot = torch.cat((x0.unsqueeze(2), one_hot_encoding), dim=2)
        
        V0_B = model.supervenient_feature_network(x0)
        V1_B = model.supervenient_feature_network(x1)

        digit_0_x0_with_one_hot = x0_with_one_hot[:, 0]
        digit_1_x0_with_one_hot = x0_with_one_hot[:, 1]
        digit_2_x0_with_one_hot = x0_with_one_hot[:, 2]
        digit_3_x0_with_one_hot = x0_with_one_hot[:, 3]
        digit_4_x0_with_one_hot = x0_with_one_hot[:, 4]
        last_digit_x0_with_one_hot = x0_with_one_hot[:, 5]

        psi = 0

        psi += estimate_mutual_information('smile', V0_B, V1_B, model.critic_g, **kwargs)

        # psi -= max(estimate_mutual_information('smile', V1, digit_0_x0_with_one_hot, model.critic_h, **kwargs), 0)
        # psi -= max(estimate_mutual_information('smile', V1, digit_1_x0_with_one_hot, model.critic_h, **kwargs), 0)
        # psi -= max(estimate_mutual_information('smile', V1, digit_2_x0_with_one_hot, model.critic_h, **kwargs), 0)
        # psi -= max(estimate_mutual_information('smile', V1, digit_3_x0_with_one_hot, model.critic_h, **kwargs), 0)
        # psi -= max(estimate_mutual_information('smile', V1, digit_4_x0_with_one_hot, model.critic_h, **kwargs), 0)
        # psi -= max(estimate_mutual_information('smile', V1, last_digit_x0_with_one_hot, model.critic_h, **kwargs), 0)

        V0_A = supervenient_A_network(x0)

        print(V0_A.shape)
        print(V0_B.shape)

        CKA = CKA_function(V0_A, V0_B)

        cka_loss = CKA_weight * CKA   # multiply by CKA loss weight 

        loss = - psi + cka_loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({'MI between V0 and V1': psi.item()})
        wandb.log({'CKA': CKA.item()})
        wandb.log({'Loss': loss.item()})

    torch.save(model.state_dict(), wandb.run.dir + "/model.pt")

    wandb.finish()

    return model


def train_correlated_bit_critic(dataloader, **kwargs):
    critic = SeparableCritic(1, 1)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project="training-emergent-features", id=run_id)

    # init optimizer
    opt = optim.Adam(critic.parameters(), lr=1e-4)

    # track gradients in our three models to make sure they are being updated
    wandb.watch(critic)

    for batch in dataloader: # batch is size (batch_size, 2, 6)
        x0 = batch[:,0] # (batch_size, 6)
        x1 = batch[:,1] # (batch_size, 6)

        extra_bit_0 = x0[:, -1].unsqueeze(1) # (batch_size, 1)
        extra_bit_1 = x1[:, -1].unsqueeze(1) # (batch_size, 1)

        # zero out gradients
        opt.zero_grad()

        # compute loss
        loss = - estimate_MI_smile(extra_bit_0, extra_bit_1, critic, clip=5)

        # backprop
        loss.backward()
        opt.step()

        wandb.log({'Loss': loss.item()})

    wandb.finish()

    return critic


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

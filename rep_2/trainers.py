import torch as t
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datetime import datetime
from models import (SupervenientFeatureNetwork,
                    PredSeparableCritic, 
                    MarginalSeparableCritic,
                    MarginalSeperableCriticExpanded,
                    DifferentRepCritic)
from models import Decoder
import torch.nn as nn
from tqdm import tqdm
from SMILE_estimator import estimate_MI_smile


# TODO bro optomize this, you are calling f so many times lmaooo
def get_sum_downward_causation_terms(marginal_critic, x_one_hot, V1_B):
    """
    Sigma_i I( X_i_t0 ; V_t1_B)
    X_i_t0 is (batch_size, 6, 8) because 
    """
    sum_margininal_MIs = 0
    for i in range(6):
        scores = marginal_critic(x_one_hot[:,i], V1_B)
        sum_margininal_MIs += estimate_MI_smile(scores)

    return sum_margininal_MIs

def get_V1_AB_MI(marginal_critic, V1_A, V1_B):
    """
    Finds I( V1_A ; V1_B )
    V1_A is (batch_size, 8)... 8 because value of 1-dim rep and a 7-dim one-hot
    V1_B is (batch_size, 1)
    """
    scores = marginal_critic(V1_A, V1_B)

    return estimate_MI_smile(scores)


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


# Seperate function for tracking MI using a seperate critic
def get_V1_AB_mutual_info_for_tracking(AB_critic, x, model_A, model_B):
    """
    A function that takes in two models that output representations of the 6 digits
    and returns the MI between the two representations

    args:
        AB_critic: a critic that takes in two representations and outputs a score
        x[batch_size, 6]: digits
        model_A: previously trained rep learner
        f: currently training rep learner
    """
    A_reps = model_A(x)
    B_reps = model_B(x)

    scores = AB_critic(A_reps, B_reps)

    V_AB_mutual_info = estimate_MI_smile(scores)

    return V_AB_mutual_info


def train_model_A(dataloader):

    pred_critic = PredSeparableCritic()
    marginal_critic = MarginalSeparableCritic()
    
    f = SupervenientFeatureNetwork()

    print("Pred Critic")
    print(pred_critic)
    print("Marginal Critic")
    print(marginal_critic)
    print("Supervenient Feature Network")
    print(f)

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

        one_hot_encoding = F.one_hot(t.tensor([0,1,2,3,4,5])).unsqueeze(0).repeat(batch_len, 1, 1)
        x0_with_one_hot = t.cat((x0.unsqueeze(2), one_hot_encoding), dim=2) # (batch_size, 6, 7)

        f.zero_grad()
        pred_critic.zero_grad()
        marginal_critic.zero_grad()

        # pred terms

        V_mutual_info = get_causally_decoupled_MI(pred_critic, x0, x1, f)
        g_loss = - V_mutual_info

        opt_pred.zero_grad()
        g_loss.backward(retain_graph=True)
        opt_pred.step()


        # marginal terms

        sum_margininal_MIs = get_sum_downward_causation_terms(marginal_critic, x0_with_one_hot, x1, f)
        marginal_loss = - sum_margininal_MIs

        opt_marginal.zero_grad()
        marginal_loss.backward(retain_graph=True)
        opt_marginal.step()


        # f

        psi = get_causally_decoupled_MI(pred_critic, x0, x1, f) - get_sum_downward_causation_terms(marginal_critic, x0_with_one_hot, x1, f)
        f_loss = - psi

        opt_f.zero_grad()
        f_loss.backward()
        opt_f.step()


        wandb.log({'Psi': psi.item()})
        wandb.log({'Sum of Marginal MIs': sum_margininal_MIs.item()})
        wandb.log({'V Mutual Info': V_mutual_info.item()})

    wandb.finish()

    return f


def train_unsimilar_model_smile(model_A, dataloader):

    pred_critic = PredSeparableCritic()
    marginal_critic = MarginalSeperableCriticExpanded()
    AB_critic = DifferentRepCritic()
    # model A is the previously trained model
    # we will now train model B so that it's representations have no MI with model A's representations
    model_B = SupervenientFeatureNetwork()


    print(pred_critic)
    print(marginal_critic)
    print(AB_critic)
    print(model_B)

    
    wandb.init(project="training-emergent-features", id=datetime.now().strftime("model-B-%Y%m%d-%H%M%S"))
    wandb.watch(pred_critic)
    wandb.watch(marginal_critic)
    wandb.watch(model_B)

    opt_pred = optim.Adam(pred_critic.parameters(), lr=1e-4)
    opt_marginal = optim.Adam(marginal_critic.parameters(), lr=1e-4)
    opt_model_B = optim.Adam(model_B.parameters(), lr=1e-5)
    opt_AB = optim.Adam(AB_critic.parameters(), lr=1e-4)

    for batch in tqdm(dataloader):

        x0 = batch[:,0] # (batch_size, 6)
        x1 = batch[:,1] # (batch_size, 6)

        batch_len = x1.shape[0]

        V1_A = model_A(x1) # (batch_size, 1)
        marginal_features = t.cat((x0, V1_A), dim=1) # (batch_size, 7)
        one_hot_encoding = F.one_hot(t.tensor([0,1,2,3,4,5,6])).unsqueeze(0).repeat(batch_len, 1, 1)
        # marginal features contains x0..5 and V1_A
        marginal_features_with_one_hot = t.cat((marginal_features.unsqueeze(2), one_hot_encoding), dim=2) # (batch_size, 7, 8)... 7 terms, 8 because value of 1-dim term and a 7-dim one-hot
        V1_A_with_one_hot = marginal_features_with_one_hot[:, 6] # (batch_size, 8)
        x0_with_one_hot = marginal_features_with_one_hot[:, :6] # (batch_size, 6, 8)


        model_B.zero_grad()
        pred_critic.zero_grad()
        marginal_critic.zero_grad()
        AB_critic.zero_grad()

        # pred critic

        V_mutual_info = get_causally_decoupled_MI(pred_critic, x0, x1, model_B)
        g_loss = - V_mutual_info

        opt_pred.zero_grad()
        g_loss.backward(retain_graph=True)
        opt_pred.step()


        # marginal critic

        # TODO remove hard coded values
        sum_downward_causation_terms = get_sum_downward_causation_terms(
            marginal_critic,
            x0_with_one_hot,
            V1_B = model_B(x1)
        )
        V1_AB = get_V1_AB_MI(
            marginal_critic,
            V1_A_with_one_hot,
            V1_B=model_B(x1)
        )
        marginal_term_loss = - sum_downward_causation_terms - V1_AB

        opt_marginal.zero_grad()
        marginal_term_loss.backward(retain_graph=True)
        opt_marginal.step()


        # AB critic
        # For tracking only

        V1_AB_mutual_info = get_V1_AB_mutual_info_for_tracking(AB_critic, x1, model_A, model_B)
        AB_loss = - V1_AB_mutual_info

        opt_AB.zero_grad()
        AB_loss.backward(retain_graph=True)
        opt_AB.step()


        # model B

        V_mutual_info = get_causally_decoupled_MI(pred_critic, x0, x1, model_B)
        marginal_terms_mutual_info = get_sum_downward_causation_terms(marginal_critic, x0_with_one_hot, V1_B=model_B(x1))
        different_V_mutual_info = get_V1_AB_MI(marginal_critic, V1_A_with_one_hot, V1_B=model_B(x1))

        Psi = V_mutual_info - marginal_terms_mutual_info
        model_B_loss = - (V_mutual_info - marginal_terms_mutual_info - different_V_mutual_info)

        opt_model_B.zero_grad()
        model_B_loss.backward()
        opt_model_B.step()

        wandb.log({'model b loss': model_B_loss.item()})
        wandb.log({'Psi': Psi.item()})
        wandb.log({'Sum of Marginal MIs': marginal_terms_mutual_info.item()})
        wandb.log({'I(VB_0; VB_1)': V_mutual_info.item()})
        wandb.log({'I(VA_1; VB_1)': different_V_mutual_info.item()})


    wandb.finish()

    return model_B







# def CKA_function(X, Y):
#     assert X.shape[0] == Y.shape[0]
#     X = X - torch.mean(X, dim=0)
#     Y = Y - torch.mean(Y, dim=0)

#     XTX = X.t() @ X
#     YTY = Y.t() @ Y
#     XTY = X.t() @ Y

#     result = XTY.norm(p="fro") ** 2 / (XTX.norm(p="fro") * YTY.norm(p="fro"))

#     del XTX, YTY, XTY
#     torch.cuda.empty_cache()

#     return result


def train_extra_bit_decoder(dataloader, f_network):
    """Main training loop that estimates time-varying MI."""

    num_steps = 10000

    model = Decoder().to('cpu')


    # track gradients in our three models to make sure they are being updated

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

    return model


def train_parity_bit_decoder(dataloader, f_network):
    """Main training loop that estimates time-varying MI."""

    num_steps = 10000

    model = Decoder().to('cpu')


    opt = optim.Adam(model.parameters(), lr=1e-4)

    counter = 0
    for batch in dataloader:
        counter += 1
        if counter > num_steps:
            break

        x1 = batch[:,1]

        parity_batch = t.sum(x1[:,:5], dim=1) % 2

        V1 = f_network(x1)

        loss = nn.MSELoss()

        # compute loss
        loss = loss(model(V1), parity_batch.unsqueeze(1))

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model

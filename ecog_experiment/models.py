import torch
import torch.nn as nn
from SMILE_estimator import estimate_MI_smile
from einops import rearrange, reduce, repeat



# This is just straight up broken it seems lol

class DecoupledCausationMI(nn.Module):
    def __init__(self):
        super(DownwardCausationMI, self).__init__()
    
    def get_MI(self):
        raise NotImplementedError("This is an abstract class")
    
    def learning_loss(self):
        raise NotImplementedError("This is an abstract class")


class DownwardCausationMI(nn.Module):
    def __init__(self):
        super(DownwardCausationMI, self).__init__()
    
    def get_singlge_marginal_MI(self):
        raise NotImplementedError("This is an abstract class")
    
    def get_sum_MI(self):
        raise NotImplementedError("This is an abstract class")
    
    def learning_loss(self):
        raise NotImplementedError("This is an abstract class")

    
class SupervenientEncoder(nn.Module):
    def __init__(self, num_features, V_dim):
        super(SupervenientEncoder, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, V_dim)
        )

    def forward(self, x):
        return self.f(x)
    
    def learning_loss(self, X0, X1, downward: DownwardCausationMI, decoupled: DecoupledCausationMI):

        V0 = self.forward(X0)
        V1 = self.forward(X1)

        decoupled_MI = decoupled.get_MI(V0, V1)
        downward_MI = downward.get_sum_MI(X0, V1)

        return - (decoupled_MI - downward_MI) + 0.01 * torch.norm(self.f[0].weight, p=2) ** 2
            

class DownwardSmile(DownwardCausationMI):
    # atom here refers to a constuent part of X_t, X_t_i
    def __init__(self, V_dim, critic_ouput_dim, feature_num):
        super(DownwardSmile, self).__init__()

        self.v_encoder = nn.Sequential(
            nn.Linear(V_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, critic_ouput_dim),
        )

        # atom_with_one_hot_dim should be (num_atoms + 1)
        self.xi_encoder = nn.Sequential(
            nn.Linear(( feature_num + 1 ), 128), # (feature_num+1) because we add a one hot (feature_num dim) to 1 feature
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, critic_ouput_dim),
        )
    
    def forward(self):
        return None

    def get_single_marginal_MI(self, x0i_with_oh, V1):
        # gets the MI between our emeergent featuere and a single atom
        # x0i_with_oh is (batch_size, num_atoms + 1)
        V1_encoded = self.v_encoder(V1)
        x0i_encoded = self.xi_encoder(x0i_with_oh)
        scores = torch.matmul(x0i_encoded, V1_encoded.t())
        return estimate_MI_smile(scores)

    def get_sum_MI(self, X0, V1):

        def _add_one_hot(X0):
            batch_len, num_features = X0.size()
            eye = torch.eye(num_features) # f * f
            eye_repeated = repeat(eye, 'f1 f2 -> b f1 f2', b=batch_len)
            X0_unsqueezed = rearrange(X0, 'b f -> b f 1')
            return torch.cat((X0_unsqueezed, eye_repeated), dim=2)

        X0_with_oh = _add_one_hot(X0)
        # X0_with_oh is (batch_size, num_atoms, num_atoms + 1)

        sum_margininal_MIs = 0
        _, num_features, _ = X0_with_oh.size()

        for i in range(num_features):
            x0i_with_oh = X0_with_oh[:, i]
            sum_margininal_MIs += self.get_single_marginal_MI(x0i_with_oh, V1)

        return sum_margininal_MIs

    def learning_loss(self, X0, V1):
        # X0 is (batch_size, num_atoms)
        return - self.get_sum_MI(X0, V1) + 0.1*(torch.norm(self.xi_encoder[0].weight, p=2) ** 2 + torch.norm(self.v_encoder[0].weight, p=2) ** 2)
    


class DecoupledSmile(DecoupledCausationMI):
    def __init__(self, V_dim, critic_ouput_dim):
        super(DecoupledCausationMI, self).__init__()

        self.v_encoder = nn.Sequential(
            nn.Linear(V_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, critic_ouput_dim),
        )

        self.W = nn.Linear(critic_ouput_dim, critic_ouput_dim, bias=False)

    def get_MI(self, V0, V1):
        V0_encoded = self.v_encoder(V0) # (batch_size, critic_ouput_dim)
        V1_encoded = self.W (self.v_encoder(V1)) # (batch_size, critic_ouput_dim)

        scores = torch.matmul(V0_encoded, V1_encoded.t())

        return estimate_MI_smile(scores)

    def learning_loss(self, V0, V1):
        # plus regularization term times a small constant
        return - self.get_MI(V0, V1) + 0.01 * torch.norm(self.W.weight, p=2) ** 2
    
    
    def forward(self):
        return None
    








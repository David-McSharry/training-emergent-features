from torch import nn
import torch
from einops import repeat, rearrange, reduce


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)



class BinaryCLUB(nn.Module):

    def __init__(self):

        super(BinaryCLUB, self).__init__()

        self.prob_of_1_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def learning_loss(self, x, y):

        y_pred = self.prob_of_1_net(x)

        y = rearrange(y, 'i 1 -> i')
        y_pred = rearrange(y_pred, 'i 1 -> i')
        y_pred = torch.where(y == 0, 1 - y_pred, y_pred)
        log_prob = torch.log2(y_pred)
        return - log_prob.mean()
    
    def MI(self, x, y):
        y_pred = self.prob_of_1_net(x)

        y = rearrange(y, 'i 1 -> i')
        y_pred = rearrange(y_pred, 'i 1 -> i')
        y_pred = torch.where(y == 0, 1 - y_pred, y_pred)
        # print(f"{x[0]} - {y[0]} - {y_pred[0]}")
        log_prob = torch.log2(y_pred)
        positive_term = log_prob.mean()

        y_shuffled = y[torch.randperm(y.size(0))]
        print('...')
        print(x.squeeze(1).eq(y_shuffled).sum() / x.nelement())
        print(x.squeeze(1).eq(y_shuffled)[:5])
        print(x.squeeze(1)[:6])
        print(y_shuffled[:6])
    
        y_pred = self.prob_of_1_net(x)
        y_pred = rearrange(y_pred, 'i 1 -> i')
        y_pred = torch.where(y_shuffled == 0, 1 - y_pred, y_pred)
        log_prob = torch.log2(y_pred)
        negative_term = log_prob.mean()

        return positive_term - negative_term


    
    # def MI(self, x, y):
    #     y_pred = self.prob_of_1_net(x)

    #     y = rearrange(y, 'i 1 -> i')
    #     y_pred = rearrange(y_pred, 'i 1 -> i')

    #     y_pred = repeat(y_pred, 'i -> i j', j = y_pred.size(0))
    #     y = repeat(y, 'i -> j i', j = y.size(0))
    #     mask = (y==0)
    #     y_pred = y_pred.clone()
    #     y_pred[mask] = 1 - y_pred[mask]
    #     log_prob = torch.log2(y_pred)

    #     # diag mean is positive
    #     first_term = log_prob.diag().mean()

    #     # off diag mean is negative
    #     mask = (torch.ones_like(log_prob) - torch.eye(log_prob.size(0))).to(device=log_prob.device)
    #     masked_log_prob = log_prob * mask
    #     second_term = masked_log_prob.sum() / mask.sum()

    #     return first_term - second_term


# %%
    
import torch
    
x = torch.tensor([1,2,3])
y = torch.tensor([0,0,3])

print(x.eq(y).sum() / x.nelement())
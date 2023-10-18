# https://github.com/ermongroup/smile-mi-estimator/blob/master/estimators.py#L129

import numpy as np
import torch
import torch.nn.functional as F

def logmeanexp_nodiag(x, dim=None, device='cpu'):
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)

    logsumexp = torch.logsumexp(
        x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=dim
      )

    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except ValueError:
        num_elem = batch_size - 1
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)


def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term


def smile_lower_bound(f, clip):
    f_ = torch.clamp(f, -clip, clip)

    z = logmeanexp_nodiag(f_, dim=(0, 1))
    dv = f.diag().mean() - z

    js = js_fgan_lower_bound(f)

    dv_js = dv - js

    return js + dv_js


def smile_estimate_mutual_information(x, y, critic_fn, device,
                                baseline_fn=None, alpha_logit=None, **kwargs):
    """Estimate variational lower bounds on mutual information.

  Args:
    estimator: string specifying estimator, one of:
      'nwj', 'infonce', 'tuba', 'js', 'interpolated'
    x: [batch_size, dim_x] Tensor
    y: [batch_size, dim_y] Tensor
    critic_fn: callable that takes x and y as input and outputs critic scores
      output shape is a [batch_size, batch_size] matrix
    baseline_fn (optional): callable that takes y as input 
      outputs a [batch_size]  or [batch_size, 1] vector
    alpha_logit (optional): logit(alpha) for interpolated bound

  Returns:
    scalar estimate of mutual information
    """
    x, y = x.to(device), y.to(device)
    print("x")
    print(x)
    print(x.shape)
    print('[[[[[[[[[[[[]]]]]]]]]]]]')
    print("y")
    print(y)
    print(y.shape)
    print('[[[[[[[[[[[[]]]]]]]]]]]]')
    
    scores = critic_fn(x, y)

    print("scores")
    print(scores)
    print(scores.shape)
    print('[[[[[[[[[[[[]]]]]]]]]]]]')

    mi = smile_lower_bound(scores, **kwargs)

    print("mi")
    print(mi)
    print(mi.shape)
    print('[[[[[[[[[[[[]]]]]]]]]]]]')
    
    return mi
import torch

def estimate_MI_smile(scores):
    """
    Returns the MI estimate using the SMILE estimator given the scores matrix and a clip
    """

    clip = 999999999
    
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
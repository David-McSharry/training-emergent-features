import torch
import torch.nn.functional as F
from torch import nn




def infoNCE_estimator(scores):

    denominators = scores.mean(dim=1)

    positive_samples = scores.diag()

    output = torch.log2(positive_samples / denominators).mean()

    return output
    







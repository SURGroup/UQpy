import numpy as np
import torch.nn.functional as F
from torch import nn
import torch


# class ELBO(nn.Module):
#     def __init__(self, train_size):
#         super(ELBO, self).__init__()
#         self.train_size = train_size
#
#     def forward(self, input, target, kl, beta):
#         assert not target.requires_grad
#         return F.nll_loss(input, target, reduction='mean') * self.train_size + beta * kl

# def lr_linear(epoch_num, decay_start, total_epochs, start_value):
#     if epoch_num < decay_start:
#         return start_value
#     return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)


def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl



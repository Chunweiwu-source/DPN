from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F


def entropy_loss(p):
    epsilon = 1e-5
    return -1 * torch.sum(p * torch.log(p + epsilon)) / p.size(0)
    

def l2_normalize(x):
    return F.normalize(x, dim=1)


class InvariancePropagationLoss(nn.Module):
    def __init__(self, temperature=0.05, z=8):
        super(InvariancePropagationLoss, self).__init__()
        
        self.t = temperature
        self.z = z

    def forward(self, prototypes, memory_bank, return_neighbour=False):
        pro_sim = self._exp(memory_bank.get_all_dot_products(prototypes))
        
        _, bias_indices = pro_sim.topk(k=self.z, dim=1, largest=True, sorted=True)
        
        criterion = torch.nn.MSELoss()
        pseudo_tgt = memory_bank.points[bias_indices]
        count = pseudo_tgt.size(1)
        pseudo_pro = pseudo_tgt.sum(dim=1) / count
        
        loss = criterion(pseudo_pro, prototypes)

        return loss

    def _exp(self, dot_prods):
        return torch.exp(dot_prods / self.t)
    
    
class MemoryBank(object):
    def __init__(self, n_points, device, m=0.):
        self.m = m
        self.device = device
        self.n_points = n_points
        self.points = torch.zeros(n_points, 256).to(device).detach()

    def clear(self):
        self.points = torch.zeros(self.n_points, 256).to(self.device).detach()

    def random_init_bank(self):
        stdv = 1. / math.sqrt(256/3)
        self.points = torch.rand(self.n_points, 256).mul_(2*stdv).add_(-stdv).to(self.device).detach()

    def update_points(self, points, point_indices):
        norm_points = l2_normalize(points)
        data_replace = self.m * self.points[point_indices,:] + (1-self.m) * norm_points
        self.points[point_indices,:] = l2_normalize(data_replace)

    def get_all_dot_products(self, points):
        assert len(points.size()) == 2
        return torch.matmul(points, torch.transpose(self.points, 1, 0))

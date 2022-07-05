#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch layers and losses
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn.init as init


def count_parameters(m: nn.Module, only_trainable: bool = True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)



class MixupLayer(nn.Module):
    def __init__(self, num, beta):
        # num: number of samples to mixup
        # beta: mixup weight for dirichlet distribution
        super().__init__()
        self.num = num 
        self.beta = beta 
        self.prior = torch.tensor([beta] * num).cuda()
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(self.prior)

    def forward(self, x, ratio=None):
        bs, dims = x.shape[0], list(x.shape[1:])
        assert bs % self.num==0, f'MixupLayer Error: tensor shape {bs} not divided by {self.num}'
        cs = bs//self.num  # chunk size
        new_shape = [cs, self.num] + dims 
        x = x.reshape(*new_shape)
        if ratio is None:
            distrib = self.dirichlet.sample(torch.Size([cs]))
        else:
            assert list(ratio.shape)==[cs, self.num]
            distrib = ratio.clone()

        for _ in range(len(dims)):
            distrib = distrib.unsqueeze(-1)  # to match x shape
        x = (x*distrib).sum(dim=1)  # [cs] + dims
        return x

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.prior = self.prior.to(*args, **kwargs)
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(self.prior)

    def extra_repr(self):
        return f'Bx{self.num} -> B'
        

def freeze_bn(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()  # not updating running mean/var
        module.weight.requires_grad = False  # not updating weight/bis, or alpha/beta in the paper
        module.bias.requires_grad = False

def weight_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun
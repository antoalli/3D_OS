"""
    Based on: Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows, WACV 2021
    code src: https://github.com/marco-rudolph/differnet
"""

import torch
from models.nf.freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, \
    ReversibleGraphNet, OutputNode, InputNode, Node


def get_nll_loss(z, jac):
    """ check equation 4 of the paper why this makes sense - oh and just ignore the scaling here """
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]


def get_ll(z, jac):
    nll = 0.5 * torch.sum(z ** 2, dim=(1,)) - jac
    return -1 * nll


def build_nf_head(input_dim=1024, n_coupling_blocks=8, clamp_alpha=3, fc_internal=2048, dropout=0.0):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': clamp_alpha, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': fc_internal, 'dropout': dropout}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder

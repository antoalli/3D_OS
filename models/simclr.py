import torch
import torch.nn as nn
from utils.utils import DotConfig
from models.common import build_hyperspherical_proj
from models import *
from models.classifiers import get_feature_encoder


class SimCLR(nn.Module):
    def __init__(self, config, **kwargs):
        """
        Implements SimCLR (https://arxiv.org/abs/2002.05709).
        Args:
            proj_input_dim (int): encoder output size
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            proj_output_dim (int): number of dimensions of the projected features.
        """

        super().__init__()
        if isinstance(config, dict):
            config = DotConfig(config)

        self.proj_input_dim = config.proj_input_dim
        self.proj_hidden_dim = config.proj_hidden_dim
        self.proj_output_dim = config.proj_output_dim
        self.dropout = 0.5
        self.encoder = get_feature_encoder(config)

        self.head = build_hyperspherical_proj(
                self.proj_input_dim, 
                self.proj_hidden_dim, 
                self.proj_output_dim, 
                self.dropout)

    def forward(self, input):
        feat = self.encoder(input)  # [bs, features_dim]
        proj = self.head(feat)  # [bs, proj_output_dim]

        return proj

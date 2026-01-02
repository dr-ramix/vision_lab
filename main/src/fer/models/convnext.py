import torch.nn as nn
import os
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from timm.models.layers import DropPath 


class ConvNextBlock(nn.Module):
    """
    ConvNeXt residual block.

    Applies a depthwise 7×7 convolution for spatial mixing, followed by channel-wise
    LayerNorm (NHWC), a pointwise feed-forward network with GELU activation, optional
    layer scaling, and stochastic depth. The block uses a residual connection and
    preserves the input shape.

    Args:
        dim (int): Number of input/output channels.
        droppath (float): Stochastic depth rate.
        layer_scale_init_value (float): Initial value for layer scale (gamma).

    Input shape:
        (N, C, H, W)

    Output shape:
        (N, C, H, W)
    """
    def __init__(self, dim, droppath=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise_conv  = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.layer_norm      = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(in_features=dim, out_features=dim*4)
        self.gelu            = nn.GELU()
        self.pointwise_conv2 = nn.Linear(in_features=dim*4, out_features=dim)
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

        self.droppath = DropPath(droppath) if droppath > 0. else nn.Identity()


    def forward(self, x):
        identity = x

        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
    
        x = x.permute(0, 3, 1, 2)
        x = self.droppath(x)

        output = x + identity

        return output
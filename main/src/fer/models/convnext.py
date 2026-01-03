import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from timm.models.layers import DropPath, trunc_normal_


class ConvNextBlock(nn.Module):
    """
    ConvNeXt residual block.

    Applies a depthwise 7×7 convolution for spatial mixing, followed by channel-wise
    LayerNorm (NHWC), a pointwise feed-forward network with GELU activation, optional
    layer scaling, and stochastic depth. The block uses a residual connection and
    preserves the input shape.

    Args:
        dim (int): Number of input/output channels.
        drop_path (float): Stochastic depth rate.
        layer_scale_init_value (float): Initial value for layer scale (gamma).

    Input shape:
        (N, C, H, W)

    Output shape:
        (N, C, H, W)
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise_conv  = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.layer_norm      = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(in_features=dim, out_features=dim*4)
        self.gelu            = nn.GELU()
        self.pointwise_conv2 = nn.Linear(in_features=dim*4, out_features=dim)
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

        self.droppath = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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



class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class ConvNeXtFERTiny(nn.Module):
     """
     ConvNeXt-based CNN adapted for small images.

     Uses depthwise convolutions, LayerNorm, MLP-style pointwise layers,
     LayerScale, and stochastic depth. The network is organized into
     four stages with progressive downsampling, followed by global
     average pooling and a linear classification head.

     Input:  (N, C, H, W)
     Output: (N, num_classes)
    """
    def __init__(self, in_channels = 3, num_classes=6, depths=[3, 3, 9, 3], 
                dims=[96, 192, 384, 768], drop_path_rate=0.1 , layer_scale_init_value=1e-6, head_init_scale=1.): 
        """
        Initializes the ConvNeXtFER architecture.

        Args:
            in_channels (int): Number of input image channels.
            num_classes (int): Number of output classes.
            depths (list[int]): Number of ConvNeXt blocks per stage.
            dims (list[int]): Channel dimension for each stage.
            drop_path_rate (float): Maximum stochastic depth rate (linearly scheduled).
            layer_scale_init_value (float): Initial LayerScale value per block.
            head_init_scale (float): Scaling factor for classifier initialization.
        """
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=dims[0], kernel_size=2, stride=2, padding=0, groups=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.downsample_layer_1 = nn.Sequential(
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels=dims[0], out_channels = dims[1], kernel_size=2, stride=2, padding=0, groups=1)
            )
        self.downsample_layer_2 = nn.Sequential(
                LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels=dims[1], out_channels = dims[2], kernel_size=2, stride=2, padding=0, groups=1)
            )
        self.downsample_layer_3 = nn.Sequential(
                LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels=dims[2], out_channels = dims[3], kernel_size=2, stride=2, padding=0, groups=1)
            )

        drop_rate = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.Sequential(
            ConvNextBlock(dim=dims[0], drop_path=drop_rate[cur + 0], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[0], drop_path=drop_rate[cur + 1], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[0], drop_path=drop_rate[cur + 2], layer_scale_init_value = layer_scale_init_value)
        )
        cur += depths[0]

        self.stage2 = nn.Sequential(
            ConvNextBlock(dim=dims[1], drop_path=drop_rate[cur + 0], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[1], drop_path=drop_rate[cur + 1], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[1], drop_path=drop_rate[cur + 2], layer_scale_init_value = layer_scale_init_value)
        )
        cur += depths[1]
        
        self.stage3 = nn.Sequential(
            ConvNextBlock(dim=dims[2], drop_path=drop_rate[cur + 0], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[2], drop_path=drop_rate[cur + 1], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[2], drop_path=drop_rate[cur + 2], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[2], drop_path=drop_rate[cur + 3], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[2], drop_path=drop_rate[cur + 4], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[2], drop_path=drop_rate[cur + 5], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[2], drop_path=drop_rate[cur + 6], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[2], drop_path=drop_rate[cur + 7], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[2], drop_path=drop_rate[cur + 8], layer_scale_init_value = layer_scale_init_value),
        )
        cur += depths[2]

        self.stage4 = nn.Sequential(
            ConvNextBlock(dim=dims[3], drop_path=drop_rate[cur + 0], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[3], drop_path=drop_rate[cur + 1], layer_scale_init_value = layer_scale_init_value),
            ConvNextBlock(dim=dims[3], drop_path=drop_rate[cur + 2], layer_scale_init_value = layer_scale_init_value)
        )
        cur += depths[3]

        self.final_ln = nn.LayerNorm(dims[-1], eps=1e-6) 
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample_layer_1(x)
        x = self.stage2(x)
        x = self.downsample_layer_2(x)
        x = self.stage3(x)
        x = self.downsample_layer_3(x)
        x = self.stage4(x)
        x = x.mean(dim=(2, 3))     
        x = self.final_ln(x)       
        x = self.head(x)

        return x
            
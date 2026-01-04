import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from timm.models.layers import DropPath, trunc_normal_
from torchvision.ops import StochasticDepth


class ConvNextBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise_conv  = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.layer_norm      = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, dim * 4)
        self.gelu            = nn.GELU()
        self.pointwise_conv2 = nn.Linear(dim * 4, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) \
            if layer_scale_init_value > 0 else None

        self.droppath = StochasticDepth(p=drop_path, mode="row") if drop_path > 0 else nn.Identity()

    def forward(self, x):
        identity = x

        x = self.depthwise_conv(x)          # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)           # (B, H, W, C)
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)           # (B, C, H, W)
        x = self.droppath(x)
        return x + identity


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last (NHWC) and channels_first (NCHW) like Meta code."""
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


class STN(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )

        nn.init.zeros_(self.fc[-1].weight)
        self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feat = self.localization(x).view(b, 32)
        theta = self.fc(feat).view(b, 2, 3)
        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        x_warp = F.grid_sample(x, grid, align_corners=False)
        return x_warp


class STNLayer(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 6),
        )

        nn.init.zeros_(self.fc[-1].weight)
        self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        batch, channel, height, width = x.size()
        y = self.localization(x).view(batch, -1)
        theta = self.fc(y).view(batch, 2, 3)
        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=channels, out_features=hidden, bias=False),
            nn.GELU(),
            nn.Linear(in_features=hidden, out_features=channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channel)
        y = self.fc(y).view(batch, channel, 1, 1)
        return x * y


class DotProductSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.norm(x)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scale = 1 / math.sqrt(math.sqrt(self.input_dim))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale    
        attention_weights = torch.softmax(scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)       
        output = attended_values + x
        return output, attention_weights


class EmoNeXtFERTiny(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=6,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        lambda_sa=1.0, 
    ):
        super().__init__()
        self.lambda_sa = lambda_sa

        self.stn_layer = STNLayer(in_channels=in_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=dims[0], kernel_size=2, stride=2, padding=0, groups=1),  # 64->32
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )

        self.downsample_layer_1 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2, padding=0, groups=1),  # 32->16
        )
        self.downsample_layer_2 = nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2, padding=0, groups=1),  # 16->8
        )
        self.downsample_layer_3 = nn.Sequential(
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2, padding=0, groups=1),  # 8->4
        )

        self.se_layer_1 = SELayer(dims[1])
        self.se_layer_2 = SELayer(dims[2])
        self.se_layer_3 = SELayer(dims[3])

        drop_rate = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.Sequential(*[
            ConvNextBlock(dims[0], drop_path=drop_rate[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[0])
        ])
        cur += depths[0]

        self.stage2 = nn.Sequential(*[
            ConvNextBlock(dims[1], drop_path=drop_rate[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[1])
        ])
        cur += depths[1]

        self.stage3 = nn.Sequential(*[
            ConvNextBlock(dims[2], drop_path=drop_rate[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[2])
        ])
        cur += depths[2]

        self.stage4 = nn.Sequential(*[
            ConvNextBlock(dims[3], drop_path=drop_rate[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[3])
        ])

        self.final_ln = nn.LayerNorm(dims[-1], eps=1e-6)

        self.attention = DotProductSelfAttention(dims[-1])

        self.head = nn.Linear(dims[-1], num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.head.weight.data.mul_(head_init_scale)
        if self.head.bias is not None:
            self.head.bias.data.mul_(head_init_scale)

    def stn(self, x):
        return self.stn_layer(x)

    def forward(self, x, labels=None):
        # (B, C, 64, 64) -> (B, C, 64, 64)
        x = self.stn(x)

        # 64 -> 32
        x = self.stem(x)
        x = self.stage1(x)

        # 32 -> 16
        x = self.downsample_layer_1(x)
        x = self.se_layer_1(x)
        x = self.stage2(x)

        # 16 -> 8
        x = self.downsample_layer_2(x)
        x = self.se_layer_2(x)
        x = self.stage3(x)

        # 8 -> 4
        x = self.downsample_layer_3(x)
        x = self.se_layer_3(x)
        x = self.stage4(x)  # (B, 768, 4, 4)

        feat = x.mean(dim=(-2, -1))  # (B, 768)
        feat = self.final_ln(feat)
        logits = self.head(feat)

      
        if labels is not None:
            x_tokens = x.flatten(2).transpose(1, 2)  # (B, 16, 768) for 64x64
            _, attn = self.attention(x_tokens)       # (B, 16, 16)
            return logits, attn

        return logits


def compute_loss_emonext(logits, labels, attn, lambda_sa=1.0):
    mean_attention_weight = torch.mean(attn)
    attention_loss = torch.mean((attn - mean_attention_weight) ** 2)
    loss = F.cross_entropy(logits, labels, label_smoothing=0.2) + lambda_sa * attention_loss
    return loss


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lambda_sa = 0.1
    model = EmoNeXtFERTiny(in_channels=1, num_classes=7, lambda_sa=lambda_sa).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    images = torch.randn(8, 1, 64, 64, device=device)
    labels = torch.randint(0, 7, (8,), device=device)

    # TRAINING: model returns logits + attn, loss computed outside
    model.train()
    logits, attn = model(images, labels=labels)
    loss = compute_loss_emonext(logits, labels, attn, lambda_sa=lambda_sa)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # INFERENCE: model returns logits only (no attention computed)
    model.eval()
    with torch.no_grad():
        logits_inf = model(images)
        preds = logits_inf.argmax(dim=1)

    print("loss:", float(loss))
    print("preds:", preds[:5])
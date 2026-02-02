"""
EmoCatNets v0 (64x64 FER)  —  NO STN
Key defaults:
- drop_path_rate: tiny 0.15, small 0.20, base 0.25
- transformer attn_dropout: 0.0
- transformer proj_dropout: 0.10

Downsampling plan:
  stem: 64 -> 64   (no downsampling)
  down1: 64 -> 32
  down2: 32 -> 16
  down3: 16 ->  8
  stage4 (relative transformer): 8x8 tokens (64)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
from timm.layers import trunc_normal_


# -----------------------------
# Blocks (unchanged)
# -----------------------------

class ConvNextBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)  # NHWC
        self.pointwise_conv1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.pointwise_conv2 = nn.Linear(dim * 4, dim)

        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )
        self.droppath = StochasticDepth(p=drop_path, mode="row") if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """LayerNorm supporting channels_last (NHWC) and channels_first (NCHW)."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        # channels_first (N, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# -----------------------------
# Relative attention (unchanged)
# -----------------------------

class RelativePositionBias(nn.Module):
    def __init__(self, heads: int, height: int, width: int):
        super().__init__()
        self.heads = heads
        self.height = height
        self.width = width
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * height - 1) * (2 * width - 1), heads)
        )

        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, H, W)
        coords = coords.flatten(1)  # (2, T)

        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += height - 1
        relative_coords[1] += width - 1
        relative_coords[0] *= 2 * width - 1
        relative_index = relative_coords[0] + relative_coords[1]  # (T, T)

        self.register_buffer("relative_position_index", relative_index, persistent=False)
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        T = self.height * self.width
        idx = self.relative_position_index.view(-1)
        bias = self.relative_bias_table[idx]  # (T*T, heads)
        bias = bias.view(T, T, self.heads).permute(2, 0, 1).contiguous()  # (heads, T, T)
        return bias


class RelativeMHSA(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        height: int,
        width: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.rpb = RelativePositionBias(heads=heads, height=height, width=width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rpb().unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class RelativeTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        height: int,
        width: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = RelativeMHSA(
            dim=dim,
            heads=heads,
            height=height,
            width=width,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.drop_path1 = StochasticDepth(p=drop_path, mode="row") if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(proj_dropout),
        )
        self.drop_path2 = StochasticDepth(p=drop_path, mode="row") if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# -----------------------------
# Configs (unchanged)
# -----------------------------

@dataclass(frozen=True)
class EmoCatNetConfig:
    depths: Tuple[int, int, int, int]
    dims: Tuple[int, int, int, int]
    drop_path_rate: float = 0.1
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    num_heads: int = 8
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    se_reduction: int = 16


EMOCATNETS_SIZES: Dict[str, EmoCatNetConfig] = {
    "tiny":  EmoCatNetConfig(
        depths=(3, 3,  9, 2),
        dims=( 96, 192,  384,  768),
        drop_path_rate=0.10,
        num_heads=8,
        attn_dropout=0.00,
        proj_dropout=0.04,
    ),
    "small": EmoCatNetConfig(
        depths=(3, 3, 27, 2),
        dims=( 96, 192,  384,  768),
        drop_path_rate=0.15,
        num_heads=8,
        attn_dropout=0.03,
        proj_dropout=0.06,
    ),
    "base":  EmoCatNetConfig(
        depths=(3, 3, 27, 3),
        dims=(128, 256,  512, 1024),
        drop_path_rate=0.20,
        num_heads=8,
        attn_dropout=0.05,
        proj_dropout=0.10,
    ),
    "large": EmoCatNetConfig(
        depths=(3, 3, 27, 2),
        dims=(192, 384,  768, 1536),
        drop_path_rate=0.30,
        num_heads=8,
        attn_dropout=0.05,
        proj_dropout=0.10,
    ),
    "xlarge": EmoCatNetConfig(
        depths=(3, 3, 27, 2),
        dims=(256, 512, 1024, 2048),
        drop_path_rate=0.40,
        num_heads=8,
        attn_dropout=0.05,
        proj_dropout=0.10,
    ),
}


# -----------------------------
# EmoCatNets v0 (NO STN)
# -----------------------------

class EmoCatNetsV0(nn.Module):
    """
    EmoCatNets v0:
      stem (no downsampling) -> stage1 -> down1 -> stage2 -> down2 -> stage3 -> down3 -> stage4(rel) -> head

    64x64:
      stem: 64->64
      down1: 64->32
      down2: 32->16
      down3: 16->8
      stage4: 8x8 tokens
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        depths: Tuple[int, int, int, int] = (3, 3, 9, 2),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        se_reduction: int = 16,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must be length 4 (C,C,C,T) and (d0,d1,d2,d3).")

        d0, d1, d2, d3 = dims

        # Stem (NO downsampling): 64 -> 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d0, kernel_size=3, stride=1, padding=1),
            LayerNorm(d0, eps=1e-6, data_format="channels_first"),
        )

        # Downsampling: 64->32->16->8
        self.downsample_layer_1 = nn.Sequential(
            LayerNorm(d0, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(d0, d1, kernel_size=2, stride=2, padding=0),
        )
        self.downsample_layer_2 = nn.Sequential(
            LayerNorm(d1, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(d1, d2, kernel_size=2, stride=2, padding=0),
        )
        self.downsample_layer_3 = nn.Sequential(
            LayerNorm(d2, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(d2, d3, kernel_size=2, stride=2, padding=0),
        )

        # SE after each stage
        self.se1 = SELayer(d0, reduction=se_reduction)
        self.se2 = SELayer(d1, reduction=se_reduction)
        self.se3 = SELayer(d2, reduction=se_reduction)
        self.se4 = SELayer(d3, reduction=se_reduction)

        # Stochastic depth schedule across all blocks
        total_blocks = sum(depths)
        dp_rates: List[float] = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0

        # Stage 1 @64x64
        self.stage1 = nn.Sequential(*[
            ConvNextBlock(d0, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[0])
        ])
        cur += depths[0]

        # Stage 2 @32x32
        self.stage2 = nn.Sequential(*[
            ConvNextBlock(d1, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[1])
        ])
        cur += depths[1]

        # Stage 3 @16x16
        self.stage3 = nn.Sequential(*[
            ConvNextBlock(d2, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[2])
        ])
        cur += depths[2]

        # Stage 4 @8x8 tokens (64)
        self.stage4 = nn.Sequential(*[
            RelativeTransformerBlock(
                dim=d3,
                heads=num_heads,
                height=8,
                width=8,
                mlp_ratio=4.0,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                drop_path=dp_rates[cur + i],
            )
            for i in range(depths[3])
        ])

        self.final_ln = nn.LayerNorm(d3, eps=1e-6)
        self.head = nn.Linear(d3, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.head.weight.data.mul_(head_init_scale)
        if self.head.bias is not None:
            self.head.bias.data.mul_(head_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)          # (B, d0, 64, 64)
        x = self.stage1(x)
        x = self.se1(x)

        x = self.downsample_layer_1(x)  # (B, d1, 32, 32)
        x = self.stage2(x)
        x = self.se2(x)

        x = self.downsample_layer_2(x)  # (B, d2, 16, 16)
        x = self.stage3(x)
        x = self.se3(x)

        x = self.downsample_layer_3(x)  # (B, d3, 8, 8)

        b, c, h, w = x.shape
        if h != 8 or w != 8:
            raise ValueError(f"Expected 8x8 before stage4, got {h}x{w}. Check input size/downsampling.")

        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, d3)
        tokens = self.stage4(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = self.se4(x)

        feat = x.mean(dim=(-2, -1))
        feat = self.final_ln(feat)
        return self.head(feat)


def emocatnets_v0_fer(
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    *,
    drop_path_rate: Optional[float] = None,
    layer_scale_init_value: Optional[float] = None,
    head_init_scale: Optional[float] = None,
    num_heads: Optional[int] = None,
    attn_dropout: Optional[float] = None,
    proj_dropout: Optional[float] = None,
) -> EmoCatNetsV0:
    size = size.lower()
    if size not in EMOCATNETS_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(EMOCATNETS_SIZES.keys())}")

    cfg = EMOCATNETS_SIZES[size]
    return EmoCatNetsV0(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=cfg.depths,
        dims=cfg.dims,
        drop_path_rate=cfg.drop_path_rate if drop_path_rate is None else drop_path_rate,
        layer_scale_init_value=cfg.layer_scale_init_value if layer_scale_init_value is None else layer_scale_init_value,
        head_init_scale=cfg.head_init_scale if head_init_scale is None else head_init_scale,
        num_heads=cfg.num_heads if num_heads is None else num_heads,
        attn_dropout=cfg.attn_dropout if attn_dropout is None else attn_dropout,
        proj_dropout=cfg.proj_dropout if proj_dropout is None else proj_dropout,
        se_reduction=cfg.se_reduction,
    )


# -----------------------------
# Quick tests
# -----------------------------

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def _shape_test(image_size: int = 64, batch_size: int = 2, num_classes: int = 6, in_channels: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in EMOCATNETS_SIZES.keys():
        model = emocatnets_v0_fer(name, in_channels=in_channels, num_classes=num_classes).to(device).eval()
        x = torch.randn(batch_size, in_channels, image_size, image_size, device=device)
        y = model(x)
        print(f"{name:6s} | params={_count_params(model):,} | out={tuple(y.shape)}")
        assert y.shape == (batch_size, num_classes)


if __name__ == "__main__":
    _shape_test()

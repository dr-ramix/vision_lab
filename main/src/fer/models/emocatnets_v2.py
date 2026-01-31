"""
EmoCatNets-v2 (64x64 FER) — plain STN + plain stem (NO residual STN, NO residual stem)

This is the "main v2" you want to register as `emocatnetsv2`:
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
from timm.layers import trunc_normal_


# ============================================================
# Core Blocks
# ============================================================

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


class ConvNextBlock(nn.Module):
    """ConvNeXt residual block."""
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
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)  # NHWC
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NCHW
        x = self.droppath(x)
        return x + identity


# ============================================================
# STN (plain, not residual)
# ============================================================

class STNLayer(nn.Module):
    """Light STN. Bias initialized to identity affine; last layer weights zero."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        y = self.localization(x).view(b, -1)
        theta = self.fc(y).view(b, 2, 3)
        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


# ============================================================
# CBAM
# ============================================================

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = self.mlp(avg) + self.mlp(mx)
        ca = self.sigmoid(ca).view(b, c, 1, 1)
        x = x * ca

        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values
        sa = torch.cat([avg_map, max_map], dim=1)
        sa = self.sigmoid(self.spatial(sa))
        x = x * sa
        return x


# ============================================================
# Relative Position Bias + Relative MHSA
# ============================================================

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
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords = coords.flatten(1)

        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += height - 1
        relative_coords[1] += width - 1
        relative_coords[0] *= 2 * width - 1
        relative_index = relative_coords[0] + relative_coords[1]

        self.register_buffer("relative_position_index", relative_index, persistent=False)
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        T = self.height * self.width
        idx = self.relative_position_index.view(-1)
        bias = self.relative_bias_table[idx]
        bias = bias.view(T, T, self.heads).permute(2, 0, 1).contiguous()
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


# ============================================================
# Conv Positional Encoding (CPE) + Transformer block
# ============================================================

class ConvPosEnc(nn.Module):
    """Depthwise 3x3 conv positional encoding for (B, T, C) tokens on fixed HxW."""
    def __init__(self, dim: int, height: int, width: int):
        super().__init__()
        self.h, self.w = height, width
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        x2d = x.transpose(1, 2).reshape(b, c, self.h, self.w)
        x2d = x2d + self.dwconv(x2d)
        return x2d.flatten(2).transpose(1, 2)


class RelativeTransformerBlockV2(nn.Module):
    """Pre-LN + (CPE + Rel-MHSA) + MLP, with StochasticDepth."""
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
        self.cpe = ConvPosEnc(dim, height, width)

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
        x = self.cpe(x)
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ============================================================
# Plain stem (NOT residual), NO downsampling: 64 -> 64
# ============================================================

class PlainStem(nn.Module):
    """Plain stem: Conv3x3(s1) -> LN(ch_first)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


# ============================================================
# Config / Sizes  (RENAMED TO V2)
# ============================================================

@dataclass(frozen=True)
class EmoCatNetV2Config:
    depths: Tuple[int, int, int, int]
    dims: Tuple[int, int, int, int]
    drop_path_rate: float = 0.15
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    num_heads: int = 8
    attn_dropout: float = 0.05
    proj_dropout: float = 0.10
    cbam_reduction: int = 16


EMOCATNETS_V2_SIZES: Dict[str, EmoCatNetV2Config] = {
    "tiny":  EmoCatNetV2Config(
        depths=(2, 2,  6, 2),
        dims=( 96, 192, 384, 512),
        drop_path_rate=0.15,
        num_heads=8,
        attn_dropout=0.05,
        proj_dropout=0.10,
    ),
    "small": EmoCatNetV2Config(
        depths=(3, 3,  9, 2),
        dims=( 96, 192, 384, 640),
        drop_path_rate=0.20,
        num_heads=8,
        attn_dropout=0.05,
        proj_dropout=0.10,
    ),
    "base":  EmoCatNetV2Config(
        depths=(3, 3, 12, 3),
        dims=(128, 256, 512, 768),
        drop_path_rate=0.25,
        num_heads=8,
        attn_dropout=0.05,
        proj_dropout=0.10,
    ),
    "large": EmoCatNetV2Config(
        depths=(3, 3, 18, 3),
        dims=(192, 384, 768, 1024),
        drop_path_rate=0.30,
        num_heads=8,
        attn_dropout=0.05,
        proj_dropout=0.10,
    ),
}


# ============================================================
# EmoCatNets-v2 Model (RENAMED CLASS)
# ============================================================

class EmoCatNetsV2(nn.Module):
    """
    EmoCatNets-v2 (plain):
      (optional) plain STN -> plain stem(64->64) -> stage1(C@64)
      -> down1(64->32) -> stage2(C@32)
      -> down2(32->16) -> stage3(C@16) [save feat16]
      -> down3(16->8)  -> stage4(T@8 tokens) [save feat8]
      -> multi-scale head: concat(GAP16, GAP8) -> LN -> Linear
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 512),
        drop_path_rate: float = 0.15,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        use_stn: bool = True,
        stn_hidden: int = 32,
        cbam_reduction: int = 16,
        num_heads: int = 8,
        attn_dropout: float = 0.05,
        proj_dropout: float = 0.10,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must be length 4: (C,C,C,T) and (d0,d1,d2,d3).")

        d0, d1, d2, d3 = dims

        self.use_stn = use_stn
        self.stn = STNLayer(in_channels=in_channels, hidden=stn_hidden) if use_stn else nn.Identity()

        self.stem = PlainStem(in_channels=in_channels, out_channels=d0)

        self.down1 = nn.Sequential(
            LayerNorm(d0, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(d0, d1, kernel_size=2, stride=2, padding=0),
        )
        self.down2 = nn.Sequential(
            LayerNorm(d1, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(d1, d2, kernel_size=2, stride=2, padding=0),
        )
        self.down3 = nn.Sequential(
            LayerNorm(d2, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(d2, d3, kernel_size=2, stride=2, padding=0),
        )

        total_blocks = sum(depths)
        dp_rates: List[float] = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0

        self.stage1 = nn.Sequential(*[
            ConvNextBlock(d0, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[0])
        ])
        cur += depths[0]

        self.stage2 = nn.Sequential(*[
            ConvNextBlock(d1, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[1])
        ])
        cur += depths[1]

        self.stage3 = nn.Sequential(*[
            ConvNextBlock(d2, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[2])
        ])
        cur += depths[2]

        self.stage4 = nn.Sequential(*[
            RelativeTransformerBlockV2(
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

        self.cbam1 = CBAM(d0, reduction=cbam_reduction)
        self.cbam2 = CBAM(d1, reduction=cbam_reduction)
        self.cbam3 = CBAM(d2, reduction=cbam_reduction)
        self.cbam4 = CBAM(d3, reduction=cbam_reduction)

        self.final_ln = nn.LayerNorm(d2 + d3, eps=1e-6)
        self.head = nn.Linear(d2 + d3, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.head.weight.data.mul_(head_init_scale)
        if self.head.bias is not None:
            self.head.bias.data.mul_(head_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stn(x) if self.use_stn else x
        x = self.stem(x)

        x = self.stage1(x)
        x = self.cbam1(x)

        x = self.down1(x)
        x = self.stage2(x)
        x = self.cbam2(x)

        x = self.down2(x)
        x = self.stage3(x)
        x = self.cbam3(x)
        feat_16 = x.mean(dim=(-2, -1))

        x = self.down3(x)
        b, c, h, w = x.shape
        if h != 8 or w != 8:
            raise ValueError(f"Expected 8x8 before transformer stage4, got {h}x{w}.")

        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.stage4(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = self.cbam4(x)
        feat_8 = x.mean(dim=(-2, -1))

        feat = torch.cat([feat_16, feat_8], dim=1)
        feat = self.final_ln(feat)
        return self.head(feat)


# ============================================================
# Factory (RENAMED)
# ============================================================

def emocatnetsv2_fer(
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    *,
    use_stn: bool = True,
    drop_path_rate: Optional[float] = None,
    layer_scale_init_value: Optional[float] = None,
    head_init_scale: Optional[float] = None,
    num_heads: Optional[int] = None,
    attn_dropout: Optional[float] = None,
    proj_dropout: Optional[float] = None,
    cbam_reduction: Optional[int] = None,
    stn_hidden: int = 32,
) -> EmoCatNetsV2:
    size = size.lower()
    if size not in EMOCATNETS_V2_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(EMOCATNETS_V2_SIZES.keys())}")

    cfg = EMOCATNETS_V2_SIZES[size]
    return EmoCatNetsV2(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=cfg.depths,
        dims=cfg.dims,
        drop_path_rate=cfg.drop_path_rate if drop_path_rate is None else drop_path_rate,
        layer_scale_init_value=cfg.layer_scale_init_value if layer_scale_init_value is None else layer_scale_init_value,
        head_init_scale=cfg.head_init_scale if head_init_scale is None else head_init_scale,
        use_stn=use_stn,
        stn_hidden=stn_hidden,
        cbam_reduction=cfg.cbam_reduction if cbam_reduction is None else cbam_reduction,
        num_heads=cfg.num_heads if num_heads is None else num_heads,
        attn_dropout=cfg.attn_dropout if attn_dropout is None else attn_dropout,
        proj_dropout=cfg.proj_dropout if proj_dropout is None else proj_dropout,
    )





# ============================================================
# Quick tests
# ============================================================

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def _shape_test(image_size: int = 64, batch_size: int = 2, num_classes: int = 6, in_channels: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in EMOCATNETS_V2_SIZES.keys():
        model = emocatnetsv2_fer(name, in_channels=in_channels, num_classes=num_classes, use_stn=True).to(device).eval()
        x = torch.randn(batch_size, in_channels, image_size, image_size, device=device)
        y = model(x)
        print(f"{name:6s} | params={_count_params(model):,} | out={tuple(y.shape)}")
        assert y.shape == (batch_size, num_classes)


if __name__ == "__main__":
    _shape_test()

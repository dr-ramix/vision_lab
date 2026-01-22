"""
EmoCatNets-v3 (64x64 FER) — SLOWER DOWNSAMPLING EDIT

Now: 64 -> 64 -> 32 -> 16   (no 8x8 stage)

Changes vs your provided code:
- Removed down3 (16->8)
- Transformer stage runs at 16x16 tokens (256 tokens)
- Added proj_16 (1x1 conv) to map d2 -> d3 at 16x16 before transformer
- Multi-scale head now fuses:
    GAP(16x16 pre-transformer, d2) + GAP(16x16 post-transformer, d3)

Requires:
  pip install timm torchvision
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
    """LayerNorm supporting channels_last (NHWC) and channels_first (NCHW) like Meta's ConvNeXt code."""
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
    """
    ConvNeXt residual block:
    DWConv(7x7) -> LN (NHWC) -> Linear(4x) -> GELU -> Linear -> LayerScale -> DropPath -> Residual
    Input/Output: (B, C, H, W)
    """
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)  # expects channels_last
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


# ============================================================
# STN (Residual)
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


class ResidualSTN(nn.Module):
    """
    Conservative STN:
      x_out = (1 - alpha) * x + alpha * warp(x)
    alpha is learnable, initialized small so STN starts near-identity.
    """
    def __init__(self, in_channels: int, hidden: int = 32, alpha_init: float = 0.15):
        super().__init__()
        self.stn = STNLayer(in_channels=in_channels, hidden=hidden)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xw = self.stn(x)
        a = torch.clamp(self.alpha, 0.0, 1.0)
        return (1.0 - a) * x + a * xw


# ============================================================
# CBAM (Channel + Spatial Attention)
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
# Relative Position Bias + Relative MHSA (fixed grid size per stage init)
# ============================================================

class RelativePositionBias(nn.Module):
    def __init__(self, heads: int, height: int, width: int):
        super().__init__()
        self.heads = heads
        self.height = height
        self.width = width
        T = height * width

        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * height - 1) * (2 * width - 1), heads)
        )

        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, H, W)
        coords = coords.flatten(1)  # (2, T)

        relative_coords = coords[:, :, None] - coords[:, None, :]  # (2, T, T)
        relative_coords[0] += height - 1
        relative_coords[1] += width - 1
        relative_coords[0] *= 2 * width - 1
        relative_index = relative_coords[0] + relative_coords[1]    # (T, T)

        self.register_buffer("relative_position_index", relative_index, persistent=False)

        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        T = self.height * self.width
        idx = self.relative_position_index.view(-1)       # (T*T,)
        bias = self.relative_bias_table[idx]              # (T*T, heads)
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, T, head_dim)

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
# Conv Positional Encoding (CPE)
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
# Residual Stem (NO downsampling): 64 -> 64
# ============================================================

class ResidualStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out + identity
        out = self.norm(out)
        return out


# ============================================================
# Config / Sizes
# ============================================================

@dataclass(frozen=True)
class EmoCatNetV2Config:
    depths: Tuple[int, int, int, int]          # C, C, C, T
    dims: Tuple[int, int, int, int]            # d0..d3
    drop_path_rate: float = 0.15
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    num_heads: int = 8
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    cbam_reduction: int = 16


EMOCATNETS_V3_SIZES: Dict[str, EmoCatNetV2Config] = {
    # NOTE: transformer now runs at 16x16 tokens (256 tokens), so "d3" might be worth keeping modest.
    "tiny":  EmoCatNetV2Config(depths=(2, 2, 6, 2), dims=( 96, 192, 384,  512), drop_path_rate=0.12, num_heads=8),
    "small": EmoCatNetV2Config(depths=(3, 3, 9, 2), dims=( 96, 192, 384,  640), drop_path_rate=0.15, num_heads=8),
    "base":  EmoCatNetV2Config(depths=(3, 3, 12, 3), dims=(128, 256, 512,  768), drop_path_rate=0.20, num_heads=8),
    "large": EmoCatNetV2Config(depths=(3, 3, 18, 3), dims=(192, 384, 768, 1024), drop_path_rate=0.25, num_heads=8),
}


# ============================================================
# EmoCatNets-v3 Model (SLOWER DOWNSAMPLING: stop at 16x16)
# ============================================================

class EmoCatNetsV3(nn.Module):
    """
    EmoCatNets-v3 (edited):
      Residual STN -> residual stem(64->64) -> stage1(C@64)
      -> down1(64->32) -> stage2(C@32)
      -> down2(32->16) -> stage3(C@16) [save feat16_pre]
      -> proj_16(d2->d3 at 16) -> stage4(T@16 tokens) [save feat16_post]
      -> head: concat(feat16_pre, feat16_post) -> LN -> Linear

    Input expected: 64x64
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
        stn_hidden: int = 32,
        stn_alpha_init: float = 0.15,
        cbam_reduction: int = 16,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must be length 4: (C,C,C,T) and (d0,d1,d2,d3).")

        d0, d1, d2, d3 = dims

        # Residual STN
        self.stn = ResidualSTN(in_channels=in_channels, hidden=stn_hidden, alpha_init=stn_alpha_init)

        # Residual Stem (NO downsample): 64 -> 64
        self.stem = ResidualStem(in_channels=in_channels, out_channels=d0)

        # Downsampling: 64->32->16 (STOP)
        self.down1 = nn.Sequential(
            LayerNorm(d0, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(d0, d1, kernel_size=2, stride=2, padding=0),
        )
        self.down2 = nn.Sequential(
            LayerNorm(d1, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(d1, d2, kernel_size=2, stride=2, padding=0),
        )

        # NEW: channel projection at 16x16, no spatial downsample
        self.proj_16 = nn.Sequential(
            LayerNorm(d2, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(d2, d3, kernel_size=1, stride=1, padding=0),
        )

        # Stochastic depth schedule across all blocks
        total_blocks = sum(depths)
        dp_rates: List[float] = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0

        # Stage1 (C) @64x64
        self.stage1 = nn.Sequential(*[
            ConvNextBlock(d0, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[0])
        ])
        cur += depths[0]

        # Stage2 (C) @32x32
        self.stage2 = nn.Sequential(*[
            ConvNextBlock(d1, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[1])
        ])
        cur += depths[1]

        # Stage3 (C) @16x16
        self.stage3 = nn.Sequential(*[
            ConvNextBlock(d2, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[2])
        ])
        cur += depths[2]

        # Stage4 (T) @16x16 tokens (256 tokens)  <-- UPDATED
        self.stage4 = nn.Sequential(*[
            RelativeTransformerBlockV2(
                dim=d3,
                heads=num_heads,
                height=16,
                width=16,
                mlp_ratio=4.0,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                drop_path=dp_rates[cur + i],
            )
            for i in range(depths[3])
        ])

        # CBAM after each stage (note: cbam4 now operates at 16x16, d3)
        self.cbam1 = CBAM(d0, reduction=cbam_reduction)
        self.cbam2 = CBAM(d1, reduction=cbam_reduction)
        self.cbam3 = CBAM(d2, reduction=cbam_reduction)
        self.cbam4 = CBAM(d3, reduction=cbam_reduction)

        # Head: fuse 16x16 pre/post transformer
        self.final_ln = nn.LayerNorm(d2 + d3, eps=1e-6)
        self.head = nn.Linear(d2 + d3, num_classes)

        # Init (ConvNeXt-ish)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.head.weight.data.mul_(head_init_scale)
        if self.head.bias is not None:
            self.head.bias.data.mul_(head_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # STN (residual)
        x = self.stn(x)          # (B, Cin, 64, 64)

        # Residual Stem (64->64)
        x = self.stem(x)         # (B, d0, 64, 64)

        # Stage1 @64
        x = self.stage1(x)
        x = self.cbam1(x)

        # Down + Stage2 @32
        x = self.down1(x)        # (B, d1, 32, 32)
        x = self.stage2(x)
        x = self.cbam2(x)

        # Down + Stage3 @16
        x = self.down2(x)        # (B, d2, 16, 16)
        x = self.stage3(x)
        x = self.cbam3(x)
        feat_16_pre = x.mean(dim=(-2, -1))  # (B, d2)

        # Project channels at 16x16 (no spatial downsample)
        x = self.proj_16(x)      # (B, d3, 16, 16)

        # Tokens @16
        b, c, h, w = x.shape
        if h != 16 or w != 16:
            raise ValueError(f"Expected 16x16 before transformer stage4, got {h}x{w}. Check input size/downsampling.")

        tokens = x.flatten(2).transpose(1, 2)  # (B, 256, d3)
        tokens = self.stage4(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = self.cbam4(x)
        feat_16_post = x.mean(dim=(-2, -1))    # (B, d3)

        # Head
        feat = torch.cat([feat_16_pre, feat_16_post], dim=1)  # (B, d2+d3)
        feat = self.final_ln(feat)
        logits = self.head(feat)
        return logits


def emocatnets_v3_fer(
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
    cbam_reduction: Optional[int] = None,
    stn_hidden: int = 32,
    stn_alpha_init: float = 0.15,
) -> EmoCatNetsV3:
    size = size.lower()
    if size not in EMOCATNETS_V3_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(EMOCATNETS_V3_SIZES.keys())}")

    cfg = EMOCATNETS_V3_SIZES[size]
    return EmoCatNetsV3(
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
        cbam_reduction=cfg.cbam_reduction if cbam_reduction is None else cbam_reduction,
        stn_hidden=stn_hidden,
        stn_alpha_init=stn_alpha_init,
    )


# ============================================================
# Quick tests
# ============================================================

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def _shape_test(image_size: int = 64, batch_size: int = 2, num_classes: int = 6, in_channels: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in EMOCATNETS_V3_SIZES.keys():
        model = emocatnets_v3_fer(name, in_channels=in_channels, num_classes=num_classes).to(device).eval()
        x = torch.randn(batch_size, in_channels, image_size, image_size, device=device)
        y = model(x)
        print(f"{name:6s} | params={_count_params(model):,} | out={tuple(y.shape)}")
        assert y.shape == (batch_size, num_classes)


def _one_train_step(size: str = "tiny", image_size: int = 64, num_classes: int = 6, in_channels: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = emocatnets_v3_fer(size=size, in_channels=in_channels, num_classes=num_classes).to(device).train()

    x = torch.randn(4, in_channels, image_size, image_size, device=device)
    target = torch.randint(0, num_classes, (4,), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    loss_fn = nn.CrossEntropyLoss()

    logits = model(x)
    loss = loss_fn(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print(f"{size} train step loss:", float(loss.item()))


if __name__ == "__main__":
    _shape_test(image_size=64, batch_size=2, num_classes=6, in_channels=3)
    _one_train_step(size="tiny", image_size=64, num_classes=6, in_channels=3)

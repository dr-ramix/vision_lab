"""
EmoCatNets (64x64 FER) — C-C-C-T with RELATIVE attention in the model,
PLUS an INDEPENDENT (separate weights) PLAIN self-attention head for
PATCH COVERAGE regularization computed at 8x8.

Goal: encourage the model to "look at every part of the face" by making
attention coverage over patches close to uniform (no patch ignored).

Architecture (64x64):
  x:                 64x64
  STN:               64x64
  stem (k2,s2):      32x32
  stage1 (C):        32x32
  down1 (k2,s2):     16x16
  stage2 (C):        16x16
  down2 (k2,s2):      8x8
  stage3 (C):         8x8     <-- REG ATTENTION PROBE HERE (T=64)
  down3 (k2,s2):      4x4
  stage4 (T, rel):    4x4
  GAP -> LN -> head:  logits (B, num_classes)

Training:
  logits, reg_attn = model(x, labels=labels)  # reg_attn from 8x8 map
  loss = compute_loss_emocatnets(logits, labels, reg_attn, lambda_cov=..., label_smoothing=...)
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
from timm.layers import trunc_normal_


# ============================================================
# Core blocks (ConvNeXt)
# ============================================================

class ConvNextBlock(nn.Module):
    """
    ConvNeXt residual block:
    DWConv(7x7) -> LN (NHWC) -> Linear(4x) -> GELU -> Linear -> LayerScale -> DropPath -> Residual
    Input/Output: (N, C, H, W)
    """
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


# ============================================================
# STN + SE
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


class SELayer(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
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


# ============================================================
# Relative Attention (MODEL attention inside stage4 @ 4x4)
# ============================================================

class RelativePositionBias(nn.Module):
    """
    Learnable relative position bias for a fixed feature-map size (H, W).
    Produces (heads, T, T) where T = H*W.
    """
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

        relative_coords = coords[:, :, None] - coords[:, None, :]  # (2, T, T)
        relative_coords[0] += height - 1
        relative_coords[1] += width - 1
        relative_coords[0] *= 2 * width - 1
        relative_index = relative_coords[0] + relative_coords[1]  # (T, T)

        self.register_buffer("relative_position_index", relative_index, persistent=False)
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        T = self.height * self.width
        idx = self.relative_position_index.view(-1)       # (T*T,)
        bias = self.relative_bias_table[idx]              # (T*T, heads)
        bias = bias.view(T, T, self.heads).permute(2, 0, 1).contiguous()  # (heads, T, T)
        return bias


class RelativeMHSA(nn.Module):
    """Relative-position MHSA on tokens (B, T, C) for fixed (H, W)."""
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
        out = self.proj_drop(self.proj(out))
        return out


class RelativeTransformerBlock(nn.Module):
    """Pre-LN + Relative MHSA + MLP, with StochasticDepth."""
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
        self.attn = RelativeMHSA(dim, heads, height, width, attn_dropout, proj_dropout)
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


# ============================================================
# Independent patch self-attention regularizer head @ 8x8 (PLAIN attention)
# ============================================================

class PatchSelfAttentionReg(nn.Module):
    """
    Separate attention head used ONLY for regularization (separate weights from stage4).
    Plain scaled dot-product attention (no relative bias).

    Input:  feat (B, C, H, W)
    Output: attn (B, heads, T, T), T = H*W
    """
    def __init__(self, dim: int, heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.drop = nn.Dropout(attn_dropout)

        trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(feat.shape)}")

        b, c, h, w = feat.shape
        T = h * w

        tokens = feat.flatten(2).transpose(1, 2)  # (B, T, C)
        tokens = self.norm(tokens)

        qkv = self.qkv(tokens).view(b, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        return attn


def attention_coverage_loss(reg_attn: torch.Tensor) -> torch.Tensor:
    """
    Encourage attention to cover all patches (no patch ignored).

    reg_attn: (B, heads, T, T), rows sum to 1 due to softmax.
    Coverage distribution over patches:
      c_j = mean_i A_{i,j}   (column mean)
    Push c toward uniform (1/T).

    Uses MSE (stable).
    """
    if reg_attn.dim() != 4:
        raise ValueError(f"Expected reg_attn (B, heads, T, T), got {tuple(reg_attn.shape)}")

    A = reg_attn.mean(dim=1)      # (B, T, T)
    c = A.mean(dim=1)             # (B, T)  (column-mean over queries)

    T = c.size(-1)
    u = torch.full_like(c, 1.0 / T)
    return F.mse_loss(c, u)


# ============================================================
# EmoCatNet configs + model
# ============================================================

@dataclass(frozen=True)
class EmoCatNetConfig:
    depths: Tuple[int, int, int, int]          # C, C, C, T
    dims: Tuple[int, int, int, int]            # d0..d3
    drop_path_rate: float = 0.1
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    num_heads: int = 8
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    se_reduction: int = 16


EMOCATNETS_SIZES: Dict[str, EmoCatNetConfig] = {
    "tiny":  EmoCatNetConfig(depths=(3, 3,  9, 2), dims=( 96, 192,  384,  768), drop_path_rate=0.10, num_heads=8),
    "small": EmoCatNetConfig(depths=(3, 3, 27, 2), dims=( 96, 192,  384,  768), drop_path_rate=0.15, num_heads=8),
    "base":  EmoCatNetConfig(depths=(3, 3, 27, 2), dims=(128, 256,  512, 1024), drop_path_rate=0.20, num_heads=8),
    "large": EmoCatNetConfig(depths=(3, 3, 27, 2), dims=(192, 384,  768, 1536), drop_path_rate=0.30, num_heads=8),
}


class EmoCatNets(nn.Module):
    """
    Main model uses Relative Transformer at 4x4.
    Regularization uses separate plain attention probe at 8x8.
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
        stn_hidden: int = 32,
        se_reduction: int = 16,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        reg_attn_dropout: float = 0.0,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must be length 4 (C,C,C,T) and (d0,d1,d2,d3).")

        d0, d1, d2, d3 = dims

        self.stn = STNLayer(in_channels=in_channels, hidden=stn_hidden)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d0, kernel_size=2, stride=2, padding=0),
            LayerNorm(d0, eps=1e-6, data_format="channels_first"),
        )

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

        self.se1 = SELayer(d0, reduction=se_reduction)
        self.se2 = SELayer(d1, reduction=se_reduction)
        self.se3 = SELayer(d2, reduction=se_reduction)
        self.se4 = SELayer(d3, reduction=se_reduction)

        # IMPORTANT: reg attention probe at 8x8 => dim must match d2 (stage3 output channels)
        self.reg_attn = PatchSelfAttentionReg(dim=d2, heads=num_heads, attn_dropout=reg_attn_dropout)

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
            RelativeTransformerBlock(
                dim=d3,
                heads=num_heads,
                height=4,
                width=4,
                mlp_ratio=4.0,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                drop_path=dp_rates[cur + i],
            )
            for i in range(depths[3])
        ])

        self.final_ln = nn.LayerNorm(d3, eps=1e-6)
        self.head = nn.Linear(d3, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.head.weight.data.mul_(head_init_scale)
        if self.head.bias is not None:
            self.head.bias.data.mul_(head_init_scale)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_reg_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Inference:
          logits = model(x)

        Training / regularization:
          logits, reg_attn = model(x, labels=labels)  OR  model(x, return_reg_attn=True)

        reg_attn: (B, heads, T, T) computed at 8x8 (T=64).
        """
        need_reg_attn = (labels is not None) or return_reg_attn

        x = self.stn(x)              # (B, Cin, 64, 64)

        x = self.stem(x)             # (B, d0, 32, 32)
        x = self.stage1(x)
        x = self.se1(x)

        x = self.downsample_layer_1(x)  # (B, d1, 16, 16)
        x = self.stage2(x)
        x = self.se2(x)

        x = self.downsample_layer_2(x)  # (B, d2, 8, 8)
        x = self.stage3(x)
        x = self.se3(x)

        # REG PROBE HERE @ 8x8 (does NOT change x)
        reg_attn = None
        if need_reg_attn:
            reg_attn = self.reg_attn(x)  # (B, heads, 64, 64)

        x = self.downsample_layer_3(x)  # (B, d3, 4, 4)

        # MODEL transformer stage (relative attention) @ 4x4
        b, c, h, w = x.shape
        if h != 4 or w != 4:
            raise ValueError(f"Expected 4x4 before stage4, got {h}x{w}. Check input size/downsampling.")
        tokens = x.flatten(2).transpose(1, 2)   # (B, 16, d3)
        tokens = self.stage4(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = self.se4(x)

        feat = x.mean(dim=(-2, -1))   # GAP -> (B, d3)
        feat = self.final_ln(feat)
        logits = self.head(feat)

        if need_reg_attn:
            return logits, reg_attn
        return logits


def emocatnet_fer(
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
    reg_attn_dropout: float = 0.0,
    stn_hidden: int = 32,
) -> EmoCatNets:
    size = size.lower()
    if size not in EMOCATNETS_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(EMOCATNETS_SIZES.keys())}")

    cfg = EMOCATNETS_SIZES[size]
    return EmoCatNets(
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
        reg_attn_dropout=reg_attn_dropout,
        stn_hidden=stn_hidden,
        se_reduction=cfg.se_reduction,
    )


# ============================================================
# Loss: CE + lambda_cov * coverage_loss(reg_attn)
# ============================================================

def compute_loss_emocatnets(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reg_attn: torch.Tensor,
    *,
    lambda_cov: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    cov = attention_coverage_loss(reg_attn)
    return ce + lambda_cov * cov


# ============================================================
# Quick tests
# ============================================================

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

@torch.no_grad()
def _shape_test(image_size: int = 64, batch_size: int = 2, num_classes: int = 6, in_channels: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in EMOCATNETS_SIZES.keys():
        model = emocatnet_fer(name, in_channels=in_channels, num_classes=num_classes).to(device).eval()
        x = torch.randn(batch_size, in_channels, image_size, image_size, device=device)

        y = model(x)
        assert y.shape == (batch_size, num_classes)

        y2, reg_attn = model(x, return_reg_attn=True)
        assert y2.shape == (batch_size, num_classes)
        assert reg_attn.dim() == 4  # (B, heads, T, T) with T=64 here
        print(f"{name:6s} | params={_count_params(model):,} | out={tuple(y.shape)} | reg_attn={tuple(reg_attn.shape)}")

def _one_train_step(size: str = "tiny", image_size: int = 64, num_classes: int = 6, in_channels: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = emocatnet_fer(size=size, in_channels=in_channels, num_classes=num_classes).to(device).train()

    x = torch.randn(8, in_channels, image_size, image_size, device=device)
    labels = torch.randint(0, num_classes, (8,), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    logits, reg_attn = model(x, labels=labels)
    loss = compute_loss_emocatnets(
        logits, labels, reg_attn,
        lambda_cov=0.05,          # good starting range: 0.02 - 0.1
        label_smoothing=0.05,
    )

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print(f"{size} train step loss:", float(loss.item()))

if __name__ == "__main__":
    _shape_test(image_size=64, batch_size=2, num_classes=6, in_channels=3)
    _one_train_step(size="tiny", image_size=64, num_classes=6, in_channels=3)

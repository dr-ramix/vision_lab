import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torchvision.ops import StochasticDepth

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ============================================================
# Blocks
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
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
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
# Residual Stem (NO downsampling): 64 -> 64
# ============================================================

class ResidualStem(nn.Module):
    """
    Residual stem that keeps spatial resolution (64->64).
    If in_channels != out_channels, uses a 1x1 skip projection.

    Main:
      3x3 conv -> GELU -> 3x3 conv -> LN (channels_first)
    Skip:
      identity OR 1x1 conv
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm  = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")

        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out + identity
        out = self.norm(out)
        return out


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class ConvNeXtFERConfig:
    depths: Tuple[int, int, int, int]
    dims: Tuple[int, int, int, int]
    drop_path_rate: float = 0.1
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0


CONVNEXTFER_SIZES: Dict[str, ConvNeXtFERConfig] = {
    # Note: stage4 now runs at 8x8 (instead of 4x4), so it's heavier.
    # These are kept as your original dims ladder; if you OOM, reduce dims[3].
    "tiny":  ConvNeXtFERConfig(depths=(3, 3,  9,  3), dims=( 96, 192,  384,  768), drop_path_rate=0.10),
    "small": ConvNeXtFERConfig(depths=(3, 3, 27,  3), dims=( 96, 192,  384,  768), drop_path_rate=0.15),
    "base":  ConvNeXtFERConfig(depths=(3, 3, 27,  3), dims=(128, 256,  512, 1024), drop_path_rate=0.20),
    "large": ConvNeXtFERConfig(depths=(3, 3, 27,  3), dims=(192, 384,  768, 1536), drop_path_rate=0.30),
    "xlarge":ConvNeXtFERConfig(depths=(3, 3, 27,  3), dims=(256, 512, 1024, 2048), drop_path_rate=0.40),
}


# ============================================================
# ConvNeXtFER-v2: delayed downsampling (stop at 8x8)
# 64 -> 32 -> 16 -> 8, no stem downsampling, no 8->4
# ============================================================

class ConvNeXtFERv2(nn.Module):
    """
    ConvNeXtFERv2 (64x64):

    stem (residual, stride=1): 64 -> 64
    stage1 @64
    down1 (s=2): 64 -> 32
    stage2 @32
    down2 (s=2): 32 -> 16
    stage3 @16
    down3 (s=2): 16 -> 8
    stage4 @8    (NO further downsampling)
    GAP -> LN -> head

    This keeps more spatial detail for FER.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        depths: Tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must be length 4.")

        d0, d1, d2, d3 = dims

        # Stem: NO downsampling, residual
        self.stem = ResidualStem(in_channels=in_channels, out_channels=d0)

        # Downsampling: 64->32->16->8 (NO 8->4)
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

        # Drop path schedule
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0

        # stage1 @64
        self.stage1 = nn.Sequential(*[
            ConvNextBlock(d0, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[0])
        ])
        cur += depths[0]

        # stage2 @32
        self.stage2 = nn.Sequential(*[
            ConvNextBlock(d1, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[1])
        ])
        cur += depths[1]

        # stage3 @16
        self.stage3 = nn.Sequential(*[
            ConvNextBlock(d2, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[2])
        ])
        cur += depths[2]

        # stage4 @8 (no extra downsampling)
        self.stage4 = nn.Sequential(*[
            ConvNextBlock(d3, drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
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
        # stem @64
        x = self.stem(x)         # (B, d0, 64, 64)
        x = self.stage1(x)       # (B, d0, 64, 64)

        # 64->32
        x = self.downsample_layer_1(x)  # (B, d1, 32, 32)
        x = self.stage2(x)

        # 32->16
        x = self.downsample_layer_2(x)  # (B, d2, 16, 16)
        x = self.stage3(x)

        # 16->8
        x = self.downsample_layer_3(x)  # (B, d3, 8, 8)
        x = self.stage4(x)

        # GAP
        x = x.mean(dim=(-2, -1))
        x = self.final_ln(x)
        x = self.head(x)
        return x


def convnextfer_v2(
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    *,
    drop_path_rate: Optional[float] = None,
    layer_scale_init_value: Optional[float] = None,
    head_init_scale: Optional[float] = None,
) -> ConvNeXtFERv2:
    size = size.lower()
    if size not in CONVNEXTFER_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(CONVNEXTFER_SIZES.keys())}")

    cfg = CONVNEXTFER_SIZES[size]
    return ConvNeXtFERv2(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=cfg.depths,
        dims=cfg.dims,
        drop_path_rate=cfg.drop_path_rate if drop_path_rate is None else drop_path_rate,
        layer_scale_init_value=cfg.layer_scale_init_value if layer_scale_init_value is None else layer_scale_init_value,
        head_init_scale=cfg.head_init_scale if head_init_scale is None else head_init_scale,
    )


# -----------------------------
# Quick tests
# -----------------------------
def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

@torch.no_grad()
def _shape_test(image_size: int = 64, batch_size: int = 2, num_classes: int = 6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in CONVNEXTFER_SIZES.keys():
        model = convnextfer_v2(name, num_classes=num_classes).to(device).eval()
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        y = model(x)
        print(f"{name:6s} | params={_count_params(model):,} | out={tuple(y.shape)}")
        assert y.shape == (batch_size, num_classes)

def _one_train_step(size: str = "tiny", image_size: int = 64, num_classes: int = 6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = convnextfer_v2(size, num_classes=num_classes).to(device).train()

    x = torch.randn(4, 3, image_size, image_size, device=device)
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
    _shape_test(image_size=64, batch_size=2, num_classes=6)
    _one_train_step(size="tiny", image_size=64, num_classes=6)

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

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
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
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
        x = self.depthwise_conv(x)      # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)       # (B, H, W, C)
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)       # (B, C, H, W)
        x = self.droppath(x)

        output = x + identity

        return output


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last (NHWC) and channels_first (NCHW) like ConvNeXt."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


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


class DotProductSelfAttention(nn.Module):
    """
    Patch self-attention on tokens.
    Input:  x (B, T, C)
    Output: out (B, T, C), attn (B, T, T)
    """
    def __init__(self, dim: int, attn_dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, C), got {tuple(x.shape)}")
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scale = 1.0 / math.sqrt(self.dim)  # standard
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, T, T)
        attn = torch.softmax(scores, dim=-1)                    # (B, T, T)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)                             # (B, T, C)
        out = out + x                                           # residual
        return out, attn




@dataclass(frozen=True)
class EmoNeXtConfig:
    depths: Tuple[int, int, int, int]
    dims: Tuple[int, int, int, int]
    drop_path_rate: float = 0.1
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    attn_dropout: float = 0.0

EMONEXT_SIZES: Dict[str, EmoNeXtConfig] = {
    
    "tiny":  EmoNeXtConfig(depths=(3, 3,  9,  3), dims=( 96, 192,  384,  768)),
    "small": EmoNeXtConfig(depths=(3, 3, 27,  3), dims=( 96, 192,  384,  768)),
    "base":  EmoNeXtConfig(depths=(3, 3, 27,  3), dims=(128, 256,  512, 1024)),
    "large": EmoNeXtConfig(depths=(3, 3, 27,  3), dims=(192, 384,  768, 1536)),
    "xlarge":EmoNeXtConfig(depths=(3, 3, 27,  3), dims=(256, 512, 1024, 2048)),
}


class EmoNeXtFER(nn.Module):
    """
    Same naming/logic: STN -> stem -> stages with downsampling -> GAP+LN -> head.
    Patch-attention computed only when labels provided or return_attn=True.
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
        attn_dropout: float = 0.0,
        stn_hidden: int = 32,
    ):
        super().__init__()
        self.stn_layer = STNLayer(in_channels=in_channels, hidden=stn_hidden)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=2, stride=2, padding=0),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )

        self.downsample_layer_1 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2, padding=0),
        )
        self.downsample_layer_2 = nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2, padding=0),
        )
        self.downsample_layer_3 = nn.Sequential(
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2, padding=0),
        )

        self.se_layer_1 = SELayer(dims[1])
        self.se_layer_2 = SELayer(dims[2])
        self.se_layer_3 = SELayer(dims[3])

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.Sequential(*[
            ConvNextBlock(dims[0], drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[0])
        ])
        cur += depths[0]

        self.stage2 = nn.Sequential(*[
            ConvNextBlock(dims[1], drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[1])
        ])
        cur += depths[1]

        self.stage3 = nn.Sequential(*[
            ConvNextBlock(dims[2], drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[2])
        ])
        cur += depths[2]

        self.stage4 = nn.Sequential(*[
            ConvNextBlock(dims[3], drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[3])
        ])

        self.final_ln = nn.LayerNorm(dims[-1], eps=1e-6)
        self.attention = DotProductSelfAttention(dims[-1], attn_dropout=attn_dropout)
        self.head = nn.Linear(dims[-1], num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.head.weight.data.mul_(head_init_scale)
        if self.head.bias is not None:
            self.head.bias.data.mul_(head_init_scale)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, return_attn: bool = False):
        x = self.stn_layer(x)

        x = self.stem(x)
        x = self.stage1(x)

        x = self.downsample_layer_1(x)
        x = self.se_layer_1(x)
        x = self.stage2(x)

        x = self.downsample_layer_2(x)
        x = self.se_layer_2(x)
        x = self.stage3(x)

        x = self.downsample_layer_3(x)
        x = self.se_layer_3(x)
        x = self.stage4(x)  # (B, C, H, W)

        feature = x.mean(dim=(-2, -1))   # global average pool -> (B, C)
        feature = self.final_ln(feature)
        logits = self.head(feature)

        if (labels is not None) or return_attn:
            tokens = x.flatten(2).transpose(1, 2)     # (B, T, C)
            _, attn = self.attention(tokens)          # (B, T, T)
            return logits, attn

        return logits



def emonext_fer(
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    *,
    drop_path_rate: Optional[float] = None,
    layer_scale_init_value: Optional[float] = None,
    head_init_scale: Optional[float] = None,
    attn_dropout: Optional[float] = None,
    stn_hidden: int = 32,
) -> EmoNeXtFER:
    size = size.lower()
    if size not in EMONEXT_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(EMONEXT_SIZES.keys())}")

    cfg = EMONEXT_SIZES[size]

    return EmoNeXtFER(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=cfg.depths,
        dims=cfg.dims,
        drop_path_rate=cfg.drop_path_rate if drop_path_rate is None else drop_path_rate,
        layer_scale_init_value=cfg.layer_scale_init_value if layer_scale_init_value is None else layer_scale_init_value,
        head_init_scale=cfg.head_init_scale if head_init_scale is None else head_init_scale,
        attn_dropout=cfg.attn_dropout if attn_dropout is None else attn_dropout,
        stn_hidden=stn_hidden,
    )


# -----------------------------
# Loss (training hyperparameter lambda_sa)
# -----------------------------

def compute_loss_emonext(logits: torch.Tensor, labels: torch.Tensor, attn: torch.Tensor, lambda_sa: float = 0.1):
    """
    Cross-entropy with label smoothing + weak patch-attention variance regularizer.

    attn: (B, T, T)
    """
    mean_attention_weight = attn.mean()
    attention_loss = ((attn - mean_attention_weight) ** 2).mean()
    ce = F.cross_entropy(logits, labels, label_smoothing=0.2)
    return ce + lambda_sa * attention_loss


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = "tiny"               # tiny/small/base/large/xlarge
    lr = 3e-4                   
    weight_decay = 0.05         
    drop_path_rate = 0.1      
    lambda_sa = 0.1             # attention regularizer strength (keep small)
    label_smoothing = 0.05       # in compute_loss_emonext
    attn_dropout = 0.0          

    model = emonext_fer(
        size=size,
        in_channels=1,
        num_classes=6,
        drop_path_rate=drop_path_rate,
        attn_dropout=attn_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    images = torch.randn(8, 1, 64, 64, device=device)
    labels = torch.randint(0, 6, (8,), device=device)

    model.train()
    logits, attn = model(images, labels=labels)  # training returns attn
    loss = compute_loss_emonext(logits, labels, attn, lambda_sa=lambda_sa)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_inf = model(images)               # inference returns logits only
        preds = logits_inf.argmax(dim=1)

    print("loss:", float(loss))
    print("preds:", preds[:5])

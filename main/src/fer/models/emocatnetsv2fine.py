# fer/models/emocatnets_v2_fine.py
from __future__ import annotations

from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import timm


# ============================================================
# Your EmoCatNets-v2 implementation
# (import yours instead if it already exists in fer/models/emocatnets_v2.py)
# ============================================================
from fer.models.emocatnets_v2 import EmoCatNetsV2, EMOCATNETS_V2_SIZES  # adjust if names differ


@torch.no_grad()
def _transfer_from_timm_convnext(
    model: nn.Module,
    convnext_name: str,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    """
    Transfer standard timm ConvNeXt weights into compatible EmoCatNets-v2 parts:
      - stage1..stage3 (ConvNeXt blocks)
      - down1..down3 (LN+Conv2 stride2), shape-matched
    """
    src = timm.create_model(convnext_name, pretrained=True).to(device).eval()

    if not hasattr(src, "stages") or not hasattr(src, "downsample_layers"):
        raise RuntimeError(f"'{convnext_name}' does not expose .stages/.downsample_layers (not a timm ConvNeXt?).")

    def copy_by_shape(dst: nn.Module, srcm: nn.Module) -> int:
        dst_sd = dst.state_dict()
        src_sd = srcm.state_dict()
        out = {}
        matched = 0
        for k, v in dst_sd.items():
            if k in src_sd and src_sd[k].shape == v.shape:
                out[k] = src_sd[k]
                matched += 1
            else:
                out[k] = v
        dst.load_state_dict(out, strict=False)
        return matched

    matched = 0

    # timm ConvNeXt: downsample_layers[0]=stem, [1]=after stage0, [2]=after stage1, [3]=after stage2
    matched += copy_by_shape(model.down1, src.downsample_layers[1])
    matched += copy_by_shape(model.down2, src.downsample_layers[2])
    matched += copy_by_shape(model.down3, src.downsample_layers[3])  # may match partially or 0 if last dim differs

    def copy_stage(dst_stage: nn.Sequential, src_stage: nn.Module) -> int:
        m = 0
        dst_blocks = list(dst_stage.children())
        src_blocks = list(src_stage.blocks) if hasattr(src_stage, "blocks") else list(src_stage.children())
        for j in range(min(len(dst_blocks), len(src_blocks))):
            m += copy_by_shape(dst_blocks[j], src_blocks[j])
        return m

    matched += copy_stage(model.stage1, src.stages[0])
    matched += copy_stage(model.stage2, src.stages[1])
    matched += copy_stage(model.stage3, src.stages[2])
    # DO NOT copy stage4 (transformer)

    if verbose:
        print(f"[emocatnetsv2fine] ConvNeXt init '{convnext_name}' loaded. matched tensors={matched}")


_SIZE_TO_CONVNEXT = {
    "tiny":  "convnext_tiny",
    "small": "convnext_small",
    "base":  "convnext_base",
    "large": "convnext_large",
}


def emocatnetsv2fine_fer(
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    *,
    transfer: bool = False,
    convnext_name: Optional[str] = None,
    convnext_device: str = "cpu",
    **kwargs,
) -> EmoCatNetsV2:
    """
    Factory matching your registry signature:
      factory(num_classes: int, in_channels: int, transfer: bool) -> nn.Module

    - transfer=False: normal init
    - transfer=True : transfer standard timm ConvNeXt pretrained weights into stage1-3/down1-3
    """
    size = size.lower()
    if size not in EMOCATNETS_V2_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(EMOCATNETS_V2_SIZES.keys())}")

    cfg = EMOCATNETS_V2_SIZES[size]

    model = EmoCatNetsV2(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=cfg.depths,
        dims=cfg.dims,
        drop_path_rate=cfg.drop_path_rate,
        layer_scale_init_value=cfg.layer_scale_init_value,
        head_init_scale=cfg.head_init_scale,
        num_heads=cfg.num_heads,
        attn_dropout=cfg.attn_dropout,
        proj_dropout=cfg.proj_dropout,
        cbam_reduction=cfg.cbam_reduction,
        **kwargs,
    )

    if transfer:
        cn = convnext_name or _SIZE_TO_CONVNEXT[size]
        _transfer_from_timm_convnext(model, convnext_name=cn, device=convnext_device, verbose=True)

    return model

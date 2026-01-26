# ============================================================
# (1) fer/models/emocatnets_v2_fine.py
# ============================================================
# This file provides a factory that can optionally initialize parts of your
# EmoCatNetsV2 from a timm ConvNeXt pretrained checkpoint.
#
# It is designed to be robust against small naming differences:
# - It maps timm ConvNeXt keys -> your model keys using prefix + internal-name remaps
# - It loads only tensors that exist AND match shape
# - It prints a clear summary of how many tensors were actually loaded
#
# IMPORTANT:
# - This does NOT try to load your stem/STN/CBAM/Transformer/head; only the ConvNeXt-like
#   conv backbone parts (stage1-3 + downsample 1-3) if they exist in your model.
# - It will still work if your model uses "down1/down2/down3" OR
#   "downsample_layer_1/2/3" (common variants).
#
from __future__ import annotations

from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import timm

# Import your real model implementation
from fer.models.emocatnets_v2 import EmoCatNetsV2, EMOCATNETS_V2_SIZES  # adjust names if needed


# -----------------------------
# Utilities
# -----------------------------
def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    if next(iter(sd.keys())).startswith("module."):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _choose_prefix(dst_keys: List[str], candidates: List[str]) -> Optional[str]:
    """
    Pick the first candidate prefix that appears in dst state_dict keys.
    Returns None if none exists.
    """
    for p in candidates:
        if any(k.startswith(p) for k in dst_keys):
            return p
    return None


def _build_convnext_key_map(dst_state: Dict[str, torch.Tensor]) -> Tuple[dict, dict]:
    """
    Build mapping settings based on what prefixes actually exist in the destination model.

    Returns:
      stage_prefix_map: maps timm stage index -> destination stage prefix (e.g. "stage1.")
      down_prefix_map:  maps timm downsample index -> destination down prefix (e.g. "down1.")
    """
    dst_keys = list(dst_state.keys())

    # Try to find how your model names stages
    # Common: stage1., stage2., stage3.
    # Some people: stages.0., stages.1. ... (rare, but handle)
    stage1_prefix = _choose_prefix(dst_keys, ["stage1.", "stages.0."])
    stage2_prefix = _choose_prefix(dst_keys, ["stage2.", "stages.1."])
    stage3_prefix = _choose_prefix(dst_keys, ["stage3.", "stages.2."])

    stage_prefix_map = {}
    if stage1_prefix: stage_prefix_map[0] = stage1_prefix
    if stage2_prefix: stage_prefix_map[1] = stage2_prefix
    if stage3_prefix: stage_prefix_map[2] = stage3_prefix

    # Try to find how your model names downsample layers (between stages)
    # timm ConvNeXt: downsample_layers.1, .2, .3 correspond to 1->2, 2->3, 3->4 transitions
    # Your model might use down1/down2/down3 or downsample_layer_1/2/3.
    down1_prefix = _choose_prefix(dst_keys, ["down1.", "downsample_layer_1."])
    down2_prefix = _choose_prefix(dst_keys, ["down2.", "downsample_layer_2."])
    down3_prefix = _choose_prefix(dst_keys, ["down3.", "downsample_layer_3."])

    down_prefix_map = {}
    if down1_prefix: down_prefix_map[1] = down1_prefix
    if down2_prefix: down_prefix_map[2] = down2_prefix
    if down3_prefix: down_prefix_map[3] = down3_prefix

    return stage_prefix_map, down_prefix_map


def _remap_convnext_internal_names(k: str) -> str:
    """
    Remap timm ConvNeXt block param names to your ConvNextBlock names.

    timm block keys commonly include:
      dwconv -> your depthwise_conv
      norm -> your layer_norm
      pwconv1 -> your pointwise_conv1
      pwconv2 -> your pointwise_conv2
      gamma -> your gamma
    """
    k = k.replace(".dwconv.", ".depthwise_conv.")
    k = k.replace(".norm.", ".layer_norm.")
    k = k.replace(".pwconv1.", ".pointwise_conv1.")
    k = k.replace(".pwconv2.", ".pointwise_conv2.")
    return k


@torch.no_grad()
def transfer_from_timm_convnext(
    model: nn.Module,
    convnext_name: str,
    pretrained: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Transfer timm ConvNeXt weights into your EmoCatNetsV2 where compatible:
      - stages 0..2 -> your stage1..stage3
      - downsample_layers 1..3 -> your down1..down3 (or downsample_layer_1..3)
    Skips:
      - convnext stem (downsample_layers.0) because your stem is different
      - convnext stage3 (stages.3) because your stage4 is not ConvNeXt conv stage
      - head/norm

    Returns stats dict.
    """
    src = timm.create_model(convnext_name, pretrained=pretrained).eval()
    src_sd = _strip_module_prefix(src.state_dict())
    dst_sd = model.state_dict()

    stage_prefix_map, down_prefix_map = _build_convnext_key_map(dst_sd)

    if verbose:
        print(f"[transfer] src={convnext_name} pretrained={pretrained}")
        print(f"[transfer] detected dst stage prefixes: {stage_prefix_map}")
        print(f"[transfer] detected dst down  prefixes: {down_prefix_map}")

    updates: Dict[str, torch.Tensor] = {}
    copied = 0
    skipped_missing = 0
    skipped_shape = 0

    for k_src, v_src in src_sd.items():
        # Skip convnext stem, last stage, head/norm
        if k_src.startswith("downsample_layers.0."):
            continue
        if k_src.startswith("stages.3."):
            continue
        if k_src.startswith("head.") or k_src.startswith("norm."):
            continue

        k_dst = None

        # Map downsample layers
        # downsample_layers.{1,2,3}.X -> {down1/down2/down3}.X
        for i_src, dst_prefix in down_prefix_map.items():
            src_prefix = f"downsample_layers.{i_src}."
            if k_src.startswith(src_prefix):
                k_dst = dst_prefix + k_src[len(src_prefix):]
                break

        # Map stages
        # stages.{0,1,2}.blocks.{j}.dwconv... -> stage{1,2,3}.{j}.depthwise_conv...
        if k_dst is None:
            for stage_i, dst_prefix in stage_prefix_map.items():
                src_prefix = f"stages.{stage_i}."
                if k_src.startswith(src_prefix):
                    tail = k_src[len(src_prefix):]
                    # timm often includes "blocks." in the key path
                    tail = tail.replace("blocks.", "")
                    k_dst = dst_prefix + tail
                    k_dst = _remap_convnext_internal_names(k_dst)
                    break

        if k_dst is None:
            continue

        if k_dst not in dst_sd:
            skipped_missing += 1
            continue

        if dst_sd[k_dst].shape != v_src.shape:
            skipped_shape += 1
            continue

        updates[k_dst] = v_src
        copied += 1

    # Apply updates
    if updates:
        dst_sd.update(updates)
        model.load_state_dict(dst_sd, strict=False)

    if verbose:
        print(f"[transfer] copied={copied}  skipped_missing={skipped_missing}  skipped_shape={skipped_shape}")

        # Optional quick sanity: show a few loaded keys
        sample = list(updates.keys())[:10]
        if sample:
            print("[transfer] sample loaded keys:")
            for s in sample:
                print("   ", s)

    return dict(copied=copied, skipped_missing=skipped_missing, skipped_shape=skipped_shape)


# -----------------------------
# Factory
# -----------------------------
_SIZE_TO_CONVNEXT = {
    "tiny":  "convnext_tiny",
    "small": "convnext_small",
    "base":  "convnext_base",
    "large": "convnext_large",
    "xlarge": "convnext_xlarge",
}


def emocatnetsv2fine_fer(
    *,
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    transfer: bool = False,
    convnext_name: Optional[str] = None,
    convnext_pretrained: bool = True,
    verbose: bool = True,
    **kwargs,
) -> EmoCatNetsV2:
    """
    Drop-in factory for your registry:
      factory(num_classes: int, in_channels: int, transfer: bool) -> nn.Module

    - transfer=False: normal initialization
    - transfer=True : initialize compatible conv backbone weights from timm ConvNeXt
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
        cn = convnext_name or _SIZE_TO_CONVNEXT.get(size, "convnext_tiny")
        transfer_from_timm_convnext(
            model=model,
            convnext_name=cn,
            pretrained=convnext_pretrained,
            verbose=verbose,
        )

    return model

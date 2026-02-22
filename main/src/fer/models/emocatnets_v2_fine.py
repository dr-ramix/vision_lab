
from __future__ import annotations

from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import timm

# v2 main (plain STN + plain stem)
from fer.models.emocatnets_v2 import EmoCatNetsV2, EMOCATNETS_V2_SIZES


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
    for p in candidates:
        if any(k.startswith(p) for k in dst_keys):
            return p
    return None


def _build_convnext_key_map(dst_state: Dict[str, torch.Tensor]) -> Tuple[dict, dict]:
    """
    Detect destination prefixes so this works even if you rename downsample layers.
    Returns:
      stage_prefix_map: timm stage idx -> dst prefix
      down_prefix_map:  timm downsample idx -> dst prefix
    """
    dst_keys = list(dst_state.keys())

    stage_prefix_map = {}
    s1 = _choose_prefix(dst_keys, ["stage1.", "stages.0."])
    s2 = _choose_prefix(dst_keys, ["stage2.", "stages.1."])
    s3 = _choose_prefix(dst_keys, ["stage3.", "stages.2."])
    if s1:
        stage_prefix_map[0] = s1
    if s2:
        stage_prefix_map[1] = s2
    if s3:
        stage_prefix_map[2] = s3

    down_prefix_map = {}
    d1 = _choose_prefix(dst_keys, ["down1.", "downsample_layer_1."])
    d2 = _choose_prefix(dst_keys, ["down2.", "downsample_layer_2."])
    d3 = _choose_prefix(dst_keys, ["down3.", "downsample_layer_3."])
    if d1:
        down_prefix_map[1] = d1
    if d2:
        down_prefix_map[2] = d2
    if d3:
        down_prefix_map[3] = d3

    return stage_prefix_map, down_prefix_map


def _remap_convnext_internal_names(k: str) -> str:
    """
    timm ConvNeXt block param names -> your ConvNextBlock names.
    """
    k = k.replace(".dwconv.", ".depthwise_conv.")
    k = k.replace(".norm.", ".layer_norm.")
    k = k.replace(".pwconv1.", ".pointwise_conv1.")
    k = k.replace(".pwconv2.", ".pointwise_conv2.")
    return k


@torch.no_grad()
def transfer_convnext_into_emocatnets_v2(
    model: nn.Module,
    convnext_name: str = "convnext_tiny",
    pretrained: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    timm ConvNeXt -> your EmoCatNetsV2 (conv backbone parts only)
    """
    src = timm.create_model(convnext_name, pretrained=pretrained).eval()
    src_sd = _strip_module_prefix(src.state_dict())
    dst_sd = model.state_dict()

    stage_prefix_map, down_prefix_map = _build_convnext_key_map(dst_sd)

    if (not stage_prefix_map or not down_prefix_map) and verbose:
        print("[transfer v2] WARNING: could not reliably detect stage/down prefixes.")
        print("  stage_prefix_map:", stage_prefix_map)
        print("  down_prefix_map :", down_prefix_map)

    updates: Dict[str, torch.Tensor] = {}
    copied = 0
    skipped_missing = 0
    skipped_shape = 0

    for k_src, v_src in src_sd.items():
        # Skip ConvNeXt stem, last stage, classifier parts
        if k_src.startswith("downsample_layers.0."):
            continue
        if k_src.startswith("stages.3."):
            continue
        if k_src.startswith("head.") or k_src.startswith("norm."):
            continue

        k_dst: Optional[str] = None

        # Downsample layers: downsample_layers.{1,2,3} -> down{1,2,3} (or downsample_layer_{1,2,3})
        for i_src, dst_prefix in down_prefix_map.items():
            src_prefix = f"downsample_layers.{i_src}."
            if k_src.startswith(src_prefix):
                k_dst = dst_prefix + k_src[len(src_prefix):]
                break

        # Stages: stages.{0,1,2}.blocks.{j}.* -> stage{1,2,3}.{j}.*
        if k_dst is None:
            for stage_i, dst_prefix in stage_prefix_map.items():
                src_prefix = f"stages.{stage_i}."
                if k_src.startswith(src_prefix):
                    tail = k_src[len(src_prefix):]
                    tail = tail.replace("blocks.", "")
                    k_dst = _remap_convnext_internal_names(dst_prefix + tail)
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

    if updates:
        dst_sd.update(updates)
        model.load_state_dict(dst_sd, strict=False)

    if verbose:
        print(
            f"[emocatnets v2 transfer] convnext='{convnext_name}' pretrained={pretrained} | "
            f"copied={copied} skipped_missing={skipped_missing} skipped_shape={skipped_shape}"
        )
        sample = list(updates.keys())[:10]
        if sample:
            print("[emocatnets v2 transfer] sample loaded keys:")
            for s in sample:
                print("  loaded:", s)

    return dict(copied=copied, skipped_missing=skipped_missing, skipped_shape=skipped_shape)


# -----------------------------
# Factory
# -----------------------------
_SIZE_TO_CONVNEXT = {
    "tiny": "convnext_tiny",
    "small": "convnext_small",
    "base": "convnext_base",
    "large": "convnext_large",
    "xlarge": "convnext_xlarge",
}


def emocatnetsv2_fine_fer(
    *,
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    transfer: bool = False,
    use_stn: bool = True,
    convnext_name: Optional[str] = None,
    convnext_pretrained: bool = True,
    verbose: bool = True,
    **kwargs,
) -> EmoCatNetsV2:
    """
    Registry-friendly factory:
      factory(num_classes, in_channels, transfer) -> nn.Module

    transfer=True loads timm ConvNeXt weights into stage1-3 + down1-3 (where compatible).
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
        use_stn=use_stn,
        stn_hidden=kwargs.pop("stn_hidden", 32),
        cbam_reduction=cfg.cbam_reduction,
        num_heads=cfg.num_heads,
        attn_dropout=cfg.attn_dropout,
        proj_dropout=cfg.proj_dropout,
        **kwargs,
    )

    if transfer:
        cn = convnext_name or _SIZE_TO_CONVNEXT.get(size, "convnext_tiny")
        transfer_convnext_into_emocatnets_v2(
            model=model,
            convnext_name=cn,
            pretrained=convnext_pretrained,
            verbose=verbose,
        )

    return model


# -----------------------------
# Optional: param groups helper
# -----------------------------
def make_emocatnets_v2_param_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_mult: float = 0.1,
) -> list[dict]:
    """
    backbone (loaded): stage1-3 + down1-3 (or downsample_layer_1-3)
    rest: stn, stem, cbam*, stage4, final_ln, head
    """
    backbone, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(
            (
                "stage1", "stage2", "stage3",
                "down1", "down2", "down3",
                "downsample_layer_1", "downsample_layer_2", "downsample_layer_3",
            )
        ):
            backbone.append(p)
        else:
            rest.append(p)

    groups: list[dict] = []
    if backbone:
        groups.append({"params": backbone, "lr": base_lr * backbone_lr_mult})
    if rest:
        groups.append({"params": rest, "lr": base_lr})
    return groups

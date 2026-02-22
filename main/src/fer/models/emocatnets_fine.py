from __future__ import annotations

from typing import Optional, Dict, Tuple, List, Any

import torch
import torch.nn as nn
import timm

from fer.models.emocatnets import EmoCatNets, EMOCATNETS_SIZES


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
    Detect destination prefixes so this works even if you rename attributes.

    Returns:
      stage_prefix_map: timm stage idx -> dst prefix
      down_prefix_map : timm downsample idx -> dst prefix
    """
    dst_keys = list(dst_state.keys())

    # Stages (conv stages only)
    stage_prefix_map: dict[int, str] = {}
    s1 = _choose_prefix(dst_keys, ["stage1.", "stages.0."])
    s2 = _choose_prefix(dst_keys, ["stage2.", "stages.1."])
    s3 = _choose_prefix(dst_keys, ["stage3.", "stages.2."])
    if s1:
        stage_prefix_map[0] = s1
    if s2:
        stage_prefix_map[1] = s2
    if s3:
        stage_prefix_map[2] = s3

    # Downsample layers
    down_prefix_map: dict[int, str] = {}
    d1 = _choose_prefix(dst_keys, ["downsample_layer_1.", "down1."])
    d2 = _choose_prefix(dst_keys, ["downsample_layer_2.", "down2."])
    d3 = _choose_prefix(dst_keys, ["downsample_layer_3.", "down3."])
    if d1:
        down_prefix_map[1] = d1
    if d2:
        down_prefix_map[2] = d2
    if d3:
        down_prefix_map[3] = d3

    return stage_prefix_map, down_prefix_map


def _strip_timm_blocks_prefix(tail: str) -> str:
    """
    timm often: stages.i.blocks.j.*
    your Sequential: stageX.j.*
    """
    return tail.replace("blocks.", "")


def _remap_convnext_internal_names(k: str) -> str:
    """
    timm ConvNeXt block param names -> your ConvNextBlock names
      dwconv  -> depthwise_conv
      norm    -> layer_norm
      pwconv1 -> pointwise_conv1
      pwconv2 -> pointwise_conv2
    """
    k = k.replace(".dwconv.", ".depthwise_conv.")
    k = k.replace(".norm.", ".layer_norm.")
    k = k.replace(".pwconv1.", ".pointwise_conv1.")
    k = k.replace(".pwconv2.", ".pointwise_conv2.")
    return k


@torch.no_grad()
def transfer_convnext_into_emocatnets(
    model: nn.Module,
    convnext_name: str = "convnext_tiny",
    pretrained: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    timm ConvNeXt -> your EmoCatNets v1 (conv parts only)

    Transfers:
      - stages.0/1/2 -> stage1/2/3
      - downsample_layers.1/2/3 -> downsample_layer_1/2/3
    Skips:
      - downsample_layers.0 (stem)
      - stages.3 (last stage)
      - norm/head
    """
    src = timm.create_model(convnext_name, pretrained=pretrained).eval()
    src_sd = _strip_module_prefix(src.state_dict())
    dst_sd = model.state_dict()

    stage_prefix_map, down_prefix_map = _build_convnext_key_map(dst_sd)

    if verbose:
        print(f"[transfer v1] src='{convnext_name}' pretrained={pretrained}")
        print(f"[transfer v1] detected dst stage prefixes: {stage_prefix_map}")
        print(f"[transfer v1] detected dst down  prefixes: {down_prefix_map}")

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
        if k_src.startswith(("head.", "norm.")):
            continue

        k_dst: Optional[str] = None

        # Downsample layers: downsample_layers.{1,2,3} -> dst down prefix
        for i_src, dst_prefix in down_prefix_map.items():
            src_prefix = f"downsample_layers.{i_src}."
            if k_src.startswith(src_prefix):
                k_dst = dst_prefix + k_src[len(src_prefix):]
                break

        # Stages: stages.{0,1,2}.blocks.{j}.* -> dst stage prefix
        if k_dst is None:
            for stage_i, dst_prefix in stage_prefix_map.items():
                src_prefix = f"stages.{stage_i}."
                if k_src.startswith(src_prefix):
                    tail = _strip_timm_blocks_prefix(k_src[len(src_prefix):])
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
            f"[emocatnets v1 transfer] copied={copied} "
            f"skipped_missing={skipped_missing} skipped_shape={skipped_shape}"
        )
        sample = list(updates.keys())[:12]
        if sample:
            print("[emocatnets v1 transfer] sample loaded keys:")
            for s in sample:
                print("  loaded:", s)

    return dict(copied=copied, skipped_missing=skipped_missing, skipped_shape=skipped_shape)


# ------------------------------------------------------------
# Factory (registry-friendly)
# ------------------------------------------------------------
_SIZE_TO_CONVNEXT = {
    "tiny": "convnext_tiny",
    "small": "convnext_small",
    "base": "convnext_base",
    "large": "convnext_large",
    "xlarge": "convnext_xlarge",
}


def emocatnetsfine_fer(
    *,
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    transfer: bool = False,
    convnext_name: Optional[str] = None,
    convnext_pretrained: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> EmoCatNets:
    """
    Factory for fine-tuning/transfer.

    transfer=True loads timm ConvNeXt weights into:
      stage1-3 + downsample_layer_1-3 (where compatible).

    Note:
      This assumes your EmoCatNets dims match the chosen ConvNeXt variant:
        tiny/small -> (96,192,384,768)
        base       -> (128,256,512,1024)
      Otherwise you'll see many skipped_shape.
    """
    size = size.lower()
    if size not in EMOCATNETS_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(EMOCATNETS_SIZES.keys())}")

    cfg = EMOCATNETS_SIZES[size]
    model = EmoCatNets(
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
        se_reduction=cfg.se_reduction,
        **kwargs,
    )

    if transfer:
        cn = convnext_name or _SIZE_TO_CONVNEXT.get(size, "convnext_tiny")
        transfer_convnext_into_emocatnets(
            model=model,
            convnext_name=cn,
            pretrained=convnext_pretrained,
            verbose=verbose,
        )

    return model


# ------------------------------------------------------------
# Optional: param groups helper (recommended for transfer=True)
# ------------------------------------------------------------
def make_emocatnets_param_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_mult: float = 0.1,
) -> list[dict]:
    """
    Backbone (loaded): stage1-3 + downsample_layer_1-3 (or down1-3)
    New modules: stn, stem, se*, stage4, final_ln, head
    """
    backbone, rest = [], []
    backbone_prefixes = (
        "stage1", "stage2", "stage3",
        "downsample_layer_1", "downsample_layer_2", "downsample_layer_3",
        "down1", "down2", "down3",
    )

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(backbone_prefixes):
            backbone.append(p)
        else:
            rest.append(p)

    groups: list[dict] = []
    if backbone:
        groups.append({"params": backbone, "lr": base_lr * backbone_lr_mult})
    if rest:
        groups.append({"params": rest, "lr": base_lr})
    return groups

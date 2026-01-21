from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


# ============================================================
# Public types
# ============================================================
@dataclass(frozen=True)
class LoaderBundle:
    """
    Unified interface returned by any dataloader builder.

    train/val/test: DataLoader (or iterable)
    class_to_idx  : mapping str->int
    class_order   : list[str] aligned with indices
    num_classes   : convenience
    input_channels: best-effort inferred from a batch (may be None)
    """
    train: Any
    val: Any
    test: Any
    class_to_idx: Dict[str, int]
    class_order: list[str]
    num_classes: int
    input_channels: Optional[int] = None


LoaderBuilder = Callable[[Any, Path], Tuple[Any, list[str], Dict[str, int]]]

def _norm(name: str) -> str:
    """
    Normalize dataloader names / aliases.
    Keeps registry keys consistent.
    """
    return str(name).strip().lower().replace("-", "_")
# ============================================================
# Dataloader registry (developer-friendly)
# ============================================================
_LOADER_REGISTRY: Dict[str, LoaderBuilder] = {}


def register_dataloader(*names: str) -> Callable[[LoaderBuilder], LoaderBuilder]:
    """
    Decorator to register a dataloader builder under multiple aliases.

    Example:
        @register_dataloader("main", "default", "npy")
        def build_main(settings, images_root): ...
    """
    def _decorator(fn: LoaderBuilder) -> LoaderBuilder:
        for n in names:
            key = _norm(n)
            if not key:
                continue
            _LOADER_REGISTRY[key] = fn
        return fn
    return _decorator


def available_dataloaders() -> list[str]:
    """
    Returns sorted list of registered loader names (aliases included).
    """
    return sorted(_LOADER_REGISTRY.keys())


# ============================================================
# Default builders
# ============================================================
@register_dataloader("grey", "gray", "npy_grey", "npy_gray", "dataloader_grey")
def _build_grey(settings, images_root: Path) -> Tuple[Any, list[str], Dict[str, int]]:
    from fer.dataset.dataloaders.dataloader_grey import (
        build_dataloaders as build,
        CLASS_ORDER,
        CLASS_TO_IDX,
    )

    dls = build(
        images_root=images_root,
        batch_size=int(getattr(settings, "bs", 64)),
        num_workers=int(getattr(settings, "num_workers", 4)),
        pin_memory=bool(getattr(settings, "pin_memory", True)),
        # optional override:
        # stats_json=Path(getattr(settings, "stats_json", "")) if getattr(settings, "stats_json", "") else None,
    )
    class_to_idx = getattr(dls, "class_to_idx", None) or dict(CLASS_TO_IDX)
    return dls, list(CLASS_ORDER), dict(class_to_idx)

@register_dataloader("mixed", "colorful", "npy_mixed", "dataloader_mixed", "main", "default", "npy", "all", "total")
def _build_mixed(settings, images_root: Path) -> Tuple[Any, list[str], Dict[str, int]]:
    from fer.dataset.dataloaders.dataloader_mixed import (
        build_dataloaders as build,
        CLASS_ORDER,
        CLASS_TO_IDX,
    )

    dls = build(
        images_root=images_root,  # .../standardized
        batch_size=int(getattr(settings, "bs", 64)),
        num_workers=int(getattr(settings, "num_workers", 4)),
        pin_memory=bool(getattr(settings, "pin_memory", True)),
    )
    class_to_idx = getattr(dls, "class_to_idx", None) or dict(CLASS_TO_IDX)
    return dls, list(CLASS_ORDER), dict(class_to_idx)

@register_dataloader(
    "hist_eq",
    "hist-eq",
    "histeq",
    "histogram_equalized",
    "histogram-equalized",
    "hist_eq_mtcnn",
    "mtcnn_hist_eq",
)
def _build_mtcnn_cropped_norm(settings, images_root: Path) -> Tuple[Any, list[str], Dict[str, int]]:
    from fer.dataset.dataloaders.dataloader_hist_eq import (
        build_dataloaders as build,
        CLASS_ORDER,
        CLASS_TO_IDX,
    )

    stats_json = getattr(settings, "stats_json", "")
    stats_path = Path(stats_json) if stats_json else None

    dls = build(
        images_root=images_root,
        batch_size=int(getattr(settings, "bs", 64)),
        num_workers=int(getattr(settings, "num_workers", 4)),
        pin_memory=bool(getattr(settings, "pin_memory", True)),
        stats_json=stats_path,
    )

    class_to_idx = getattr(dls, "class_to_idx", None) or dict(CLASS_TO_IDX)
    return dls, list(CLASS_ORDER), dict(class_to_idx)


@register_dataloader("fer", "fer2013", "FER2013")
def _build_legacy(settings, images_root: Path) -> Tuple[Any, list[str], Dict[str, int]]:
    from fer.dataset.dataloaders.dataloader_fer2013 import (
        build_dataloaders as build,
        CLASS_ORDER,
        CLASS_TO_IDX,
    )

    dls = build(
        images_root=images_root,
        batch_size=int(getattr(settings, "bs", 64)),
        num_workers=int(getattr(settings, "num_workers", 4)),
    )
    class_to_idx = getattr(dls, "class_to_idx", None) or dict(CLASS_TO_IDX)
    return dls, list(CLASS_ORDER), dict(class_to_idx)

@register_dataloader(
    "fer2013_no_int",
    "fer2013_no_intensity",
    "fer2013_mtcnn",
    "fer2013_cropped",
    "npy_fer2013_no_int",
)
def _build_fer2013_no_intensity(settings, images_root: Path) -> Tuple[Any, list[str], Dict[str, int]]:
    """
    FER2013 MTCNN-cropped NPY loader (no intensity normalization)

    Uses:
      standardized/fer2013/fer2013_mtcnn_cropped/npy
      standardized/fer2013/fer2013_mtcnn_cropped/dataset_stats_train.json
    """

    from fer.dataset.dataloaders.dataloder_fer2013_no_intensity import (
        build_dataloaders as build,
        CLASS_ORDER,
        CLASS_TO_IDX,
    )

    stats_json = getattr(settings, "stats_json", "")
    stats_path = Path(stats_json) if stats_json else None

    dls = build(
        images_root=images_root,  # expected: .../standardized
        batch_size=int(getattr(settings, "bs", 64)),
        num_workers=int(getattr(settings, "num_workers", 4)),
        pin_memory=bool(getattr(settings, "pin_memory", True)),
        stats_json=stats_path,
    )

    class_to_idx = getattr(dls, "class_to_idx", None) or dict(CLASS_TO_IDX)
    return dls, list(CLASS_ORDER), dict(class_to_idx)



# ============================================================
# Public API
# ============================================================
def build_loaders(settings) -> LoaderBundle:
    """
    Main entry point used by training code.

    Rules:
      - settings.dataloader selects the builder.
      - If settings.dataloader is missing/empty -> uses "main".
      - "main" is the default (as requested).

    Developer experience:
      - To add a new loader, just define a builder and decorate with @register_dataloader(...)
      - To rename, just add/remove aliases in the decorator.
    """
    # default = main
    requested = _norm(getattr(settings, "dataloader", "main") or "main")

    images_root = Path(getattr(settings, "images_root", ""))
    if not images_root:
        raise ValueError("settings.images_root is required to build dataloaders.")
    if not images_root.exists():
        raise FileNotFoundError(f"images_root does not exist: {images_root}")

    builder = _LOADER_REGISTRY.get(requested)
    if builder is None:
        raise ValueError(
            f"Unknown dataloader '{requested}'. "
            f"Available: {available_dataloaders()}"
        )

    dls, class_order, class_to_idx = builder(settings, images_root)

    _validate_class_mapping(class_to_idx, class_order)

    in_ch = _infer_in_channels(getattr(dls, "train", None))

    return LoaderBundle(
        train=dls.train,
        val=dls.val,
        test=dls.test,
        class_to_idx=class_to_idx,
        class_order=class_order,
        num_classes=len(class_order),
        input_channels=in_ch,
    )


# ============================================================
# Internal helpers
# ============================================================
def _norm(x: str) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def _validate_class_mapping(class_to_idx: Dict[str, int], class_order: list[str]) -> None:
    if not class_order:
        raise ValueError("class_order is empty.")
    if not class_to_idx:
        raise ValueError("class_to_idx is empty.")

    missing = [c for c in class_order if c not in class_to_idx]
    if missing:
        raise ValueError(f"class_to_idx missing entries for: {missing}")

    bad = [(c, class_to_idx[c], i) for i, c in enumerate(class_order) if class_to_idx[c] != i]
    if bad:
        example = ", ".join([f"{c}: idx={idx} expected={exp}" for c, idx, exp in bad[:10]])
        raise ValueError(f"class_to_idx does not match class_order positions. Examples: {example}")


def _infer_in_channels(loader: Any) -> Optional[int]:
    """
    Best-effort: peek one batch (xb, yb) and return xb.shape[1] for BCHW tensors.
    """
    if loader is None:
        return None
    try:
        xb, _ = next(iter(loader))
        if hasattr(xb, "ndim") and xb.ndim >= 2:
            return int(xb.shape[1])
    except Exception:
        return None
    return None

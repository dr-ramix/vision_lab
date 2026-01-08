# main/src/fer/inference/hub.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_REPO_ID = "drRamix/EMO_NETS_LMU"
DEFAULT_REVISION = "main"


@dataclass(frozen=True)
class ResolvedWeights:
    """
    Resolved local paths for a given model folder (either inside the project
    or inside the HF cache).
    """
    model_folder: str          # HF folder name, e.g. "resnet50"
    folder: Path              # local folder containing files
    config_path: Path
    weights_path: Path
    format: str               # "safetensors" or "pt"


def weights_root() -> Path:
    """
    Project weights root:
      main/src/fer/inference/weights
    """
    return Path(__file__).resolve().parent / "weights"


def _is_nonempty_file(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


def _pick_config_file(folder: Path) -> Path:
    """
    config.json is recommended, but in manual setups it might be missing/empty.
    We return the path if it exists (even empty), otherwise raise in resolve_local
    only if you want to enforce it.
    """
    return folder / "config.json"


def _pick_weights_file(folder: Path) -> Tuple[Path, str]:
    """
    Prefer a non-empty safetensors file, fallback to a non-empty .pt file.

    This fixes the common situation where model.safetensors exists but is 0 bytes
    (or a placeholder), while model_state_dict.pt is the real checkpoint.
    """
    st = folder / "model.safetensors"
    pt = folder / "model_state_dict.pt"

    if _is_nonempty_file(st):
        return st, "safetensors"
    if _is_nonempty_file(pt):
        return pt, "pt"

    # Give a very actionable error
    details = []
    if st.exists():
        details.append(f"model.safetensors exists (size={st.stat().st_size} bytes)")
    else:
        details.append("model.safetensors missing")
    if pt.exists():
        details.append(f"model_state_dict.pt exists (size={pt.stat().st_size} bytes)")
    else:
        details.append("model_state_dict.pt missing")

    raise FileNotFoundError(
        f"No valid (non-empty) weights file found in {folder}.\n"
        f"Details: {', '.join(details)}\n"
        f"Expected a non-empty 'model.safetensors' or 'model_state_dict.pt'."
    )


def resolve_local(model_folder: str) -> ResolvedWeights:
    """
    Resolve weights from the project folder only (manual placement).
    NEVER downloads.

    Note: config.json may be empty in manual setups; we allow that.
    """
    root = weights_root()
    folder = root / model_folder
    cfg = _pick_config_file(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Missing weights folder for '{model_folder}'.\n"
            f"Expected folder:\n  {folder}\n"
            f"You can:\n"
            f"  1) copy the files there manually, or\n"
            f"  2) call .load(source='project') to download into the project, or\n"
            f"  3) call .load(source='cache') to use Hugging Face cache.\n"
        )

    if not cfg.exists():
        raise FileNotFoundError(
            f"Missing config.json for '{model_folder}'.\n"
            f"Expected file:\n  {cfg}\n"
            f"Expected folder:\n  {folder}\n"
            f"If you do not want to store config, create an empty config.json and defaults will be used.\n"
        )

    w, fmt = _pick_weights_file(folder)
    return ResolvedWeights(
        model_folder=model_folder,
        folder=folder,
        config_path=cfg,
        weights_path=w,
        format=fmt,
    )


def download_to_project(
    model_folder: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    force: bool = False,
) -> ResolvedWeights:
    """
    Download <repo_id>/<model_folder>/* into:
      main/src/fer/inference/weights/<model_folder>/

    Uses snapshot_download so README.md/config/weights all come together.
    """
    root = weights_root()
    root.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        f"{model_folder}/config.json",
        f"{model_folder}/model.safetensors",
        f"{model_folder}/model_state_dict.pt",
        f"{model_folder}/README.md",
        f"{model_folder}/hf_manifest.json",
    ]

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=root,                 # mirrors repo structure under root
        local_dir_use_symlinks=False,   # portable, no symlinks
        allow_patterns=allow_patterns,
        force_download=force,
    )

    return resolve_local(model_folder)


def resolve_from_cache(
    model_folder: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
) -> ResolvedWeights:
    """
    Download only the required files into the Hugging Face cache and return
    paths inside the cache. Does NOT copy into the project.
    """
    cfg = hf_hub_download(repo_id, f"{model_folder}/config.json", revision=revision)

    # Prefer safetensors, fallback to pt
    try:
        w = hf_hub_download(repo_id, f"{model_folder}/model.safetensors", revision=revision)
        fmt = "safetensors"
    except Exception:
        w = hf_hub_download(repo_id, f"{model_folder}/model_state_dict.pt", revision=revision)
        fmt = "pt"

    cfg_path = Path(cfg)
    w_path = Path(w)
    return ResolvedWeights(
        model_folder=model_folder,
        folder=cfg_path.parent,
        config_path=cfg_path,
        weights_path=w_path,
        format=fmt,
    )


def ensure_weights(
    model_folder: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    source: str = "local",  # "local" | "project" | "cache"
    force: bool = False,
) -> ResolvedWeights:
    """
    Resolve weights according to requested source:

      source="local":
        - ONLY use files that exist in the project under fer/inference/weights/
        - NEVER downloads
      source="project":
        - use local if present; otherwise download into project
      source="cache":
        - download into HF cache and load from there (no project files)
    """
    src = source.strip().lower()

    if src == "local":
        return resolve_local(model_folder)

    if src == "project":
        try:
            return resolve_local(model_folder)
        except FileNotFoundError:
            return download_to_project(
                model_folder,
                repo_id=repo_id,
                revision=revision,
                force=force,
            )

    if src == "cache":
        return resolve_from_cache(model_folder, repo_id=repo_id, revision=revision)

    raise ValueError("source must be one of: 'local', 'project', 'cache'")


def where_to_put_weights(model_folder: str) -> Path:
    """
    Convenience helper for manual users.
    """
    return weights_root() / model_folder

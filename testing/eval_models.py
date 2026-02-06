from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("/home/b/bargozideh/vision_lab/vision_lab")
SRC_DIR = PROJECT_ROOT / "main" / "src"
sys.path.insert(0, str(SRC_DIR))

from fer.models.registry import make_model  # noqa: E402

TRAIN_CLASS_ORDER = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TRAIN_CLASS_ORDER)}
NPY_EXTS = {".npy"}

HF_WEIGHTS: Dict[str, Dict[str, str]] = {
    "resnet50": {"repo_id": "lmuemonets/lmu_emonets", "filename": "resnet50/model_state_dict.pt"},
    "resnet101": {"repo_id": "lmuemonets/lmu_emonets", "filename": "resnet101/model_state_dict.pt"},
    "emocatnetsv2_small": {"repo_id": "lmuemonets/lmu_emonets", "filename": "emocatnetsv2_small/model_state_dict.pt"},
}

DEFAULT_NPY_DIR = Path("/home/b/bargozideh/vision_lab/vision_lab/testing/jaffe/npy")
DEFAULT_OUT_CSV = Path("/home/b/bargozideh/vision_lab/vision_lab/testing/results_jaffe.csv")

@dataclass(frozen=True)
class Sample:
    path: Path
    y: int
    x: Optional[torch.Tensor]

def iter_labeled_npys(images_dir: Path, *, recursive: bool) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    for cls in TRAIN_CLASS_ORDER:
        cls_dir = images_dir / cls
        if not cls_dir.exists():
            continue
        paths = cls_dir.rglob("*") if recursive else cls_dir.iterdir()
        for p in paths:
            if p.is_file() and p.suffix.lower() in NPY_EXTS:
                items.append((p, CLASS_TO_IDX[cls]))
    return sorted(items, key=lambda t: str(t[0]))

def npy_to_tensor(path: Path) -> Optional[torch.Tensor]:
    try:
        arr = np.load(path)
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            return None
        if arr.shape[0] == 3:
            chw = arr
        elif arr.shape[2] == 3:
            chw = arr.transpose(2, 0, 1)
        else:
            return None
        chw = np.ascontiguousarray(chw, dtype=np.float32)
        return torch.from_numpy(chw)
    except Exception:
        return None

class LabeledInferenceDataset(Dataset):
    def __init__(self, labeled_paths: Sequence[Tuple[Path, int]]) -> None:
        self.items = list(labeled_paths)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        path, y = self.items[idx]
        x = npy_to_tensor(path)
        return Sample(path=path, y=y, x=x)

def collate_samples(samples: List[Sample]) -> Tuple[List[Path], torch.Tensor, List[Optional[torch.Tensor]]]:
    paths = [s.path for s in samples]
    ys = torch.tensor([s.y for s in samples], dtype=torch.long)
    xs = [s.x for s in samples]
    return paths, ys, xs

def _remap_emocatnetsv2_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        nk = nk.replace("stem.0.", "stem.conv1.")
        nk = nk.replace("stem.2.", "stem.conv2.")
        nk = nk.replace("stem.3.", "stem.norm.")
        out[nk] = v
    return out

def _clean_state_dict(state: Dict[str, torch.Tensor], model_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        sd[k[7:] if k.startswith("module.") else k] = v
    if model_name == "emocatnetsv2_small":
        sd = _remap_emocatnetsv2_keys(sd)
    return sd

def load_model_from_hf(
    model_name: str,
    *,
    device: torch.device,
    num_classes: int,
    in_channels: int,
    revision: Optional[str] = None,
) -> nn.Module:
    if model_name not in HF_WEIGHTS:
        raise ValueError(f"Unknown model '{model_name}'. Allowed: {sorted(HF_WEIGHTS.keys())}")

    model = make_model(model_name, num_classes=num_classes, in_channels=in_channels)

    meta = HF_WEIGHTS[model_name]
    ckpt_path = hf_hub_download(repo_id=meta["repo_id"], filename=meta["filename"], revision=revision)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format for {model_name} (expected dict or dict['state_dict']).")

    strict = (model_name != "emocatnetsv2_small")
    missing, unexpected = model.load_state_dict(_clean_state_dict(state_dict, model_name=model_name), strict=strict)

    if model_name == "emocatnetsv2_small":
        allowed_missing = {"stem.skip.weight"}
        real_missing = sorted(set(missing) - allowed_missing)
        if real_missing:
            print("[warn] emocatnetsv2_small missing keys (showing up to 20):", real_missing[:20])
        if unexpected:
            print("[warn] emocatnetsv2_small unexpected keys (showing up to 20):", unexpected[:20])
        if not strict:
            print("[warn] emocatnetsv2_small loaded with strict=False due to key-name mismatch")

    model.to(device).eval()
    return model

@torch.no_grad()
def eval_model_accuracy(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    skip_no_face: bool,
) -> Tuple[float, int, int, int]:
    n_total = 0
    n_used = 0
    n_bad = 0
    n_correct = 0

    for _paths, ys, xs in loader:
        n_total += len(xs)

        valid = [i for i, x in enumerate(xs) if x is not None]
        n_bad += (len(xs) - len(valid))

        if valid:
            xb = torch.stack([xs[i] for i in valid], dim=0).to(device)
            yb = ys[valid].to(device)

            logits = model(xb)
            pred = torch.argmax(logits, dim=1)

            n_used += len(valid)
            n_correct += int((pred == yb).sum().item())

    denom = n_used if skip_no_face else n_total
    if denom == 0:
        return 0.0, n_total, n_used, n_bad

    acc = (n_correct / max(1, denom)) if not skip_no_face else (n_correct / max(1, n_used))
    return float(acc), n_total, n_used, n_bad

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HF FER models on labeled NPY folder dataset.")
    p.add_argument("--npy-dir", type=str, default=str(DEFAULT_NPY_DIR))
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-no-face", action="store_true")
    p.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "resnet101", "emocatnetsv2_small"],
        choices=sorted(HF_WEIGHTS.keys()),
    )
    p.add_argument("--hf-revision", type=str, default=None)
    p.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV))
    return p.parse_args()

def main() -> int:
    args = parse_args()
    npy_dir = Path(args.npy_dir).expanduser().resolve()
    if not npy_dir.exists():
        raise FileNotFoundError(f"npy-dir not found: {npy_dir}")

    labeled = iter_labeled_npys(npy_dir, recursive=args.recursive)
    if not labeled:
        raise RuntimeError(
            "No labeled .npy files found.\n"
            "Expected structure:\n"
            "  npy_dir/anger/*.npy\n"
            "  npy_dir/disgust/*.npy\n"
            "  ...\n"
            f"Classes: {TRAIN_CLASS_ORDER}\n"
            f"Got npy_dir: {npy_dir}"
        )

    device = torch.device(args.device)

    ds = LabeledInferenceDataset(labeled)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_samples,
    )

    num_classes = len(TRAIN_CLASS_ORDER)
    in_channels = 3

    print(f"[info] npy_dir:       {npy_dir}")
    print(f"[info] device:        {device}")
    print(f"[info] models:        {args.models}")
    print(f"[info] skip_no_face:  {args.skip_no_face}")
    print(f"[info] n_images:      {len(ds)}")

    results: List[Tuple[str, float, int, int, int]] = []

    for name in args.models:
        model = load_model_from_hf(
            name,
            device=device,
            num_classes=num_classes,
            in_channels=in_channels,
            revision=args.hf_revision,
        )

        acc, n_total, n_used, n_bad = eval_model_accuracy(
            model,
            loader,
            device=device,
            skip_no_face=args.skip_no_face,
        )

        denom = n_used if args.skip_no_face else n_total
        print(f"[result] {name:18s}  acc={acc*100:6.2f}%  denom={denom}  total={n_total}  used={n_used}  bad={n_bad}")
        results.append((name, acc, n_total, n_used, n_bad))

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "n_total", "n_used", "n_bad"])
        for name, acc, n_total, n_used, n_bad in results:
            w.writerow([name, f"{acc:.6f}", n_total, n_used, n_bad])
    print(f"[info] wrote summary csv: {out_path}")

    print("[done]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

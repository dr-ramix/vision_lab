from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import getpass
import json
import os
import platform
import random
import string
import time
from typing import Dict, Any

def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(n))

def get_repo_root_from_main(project_root_main: Path) -> Path:
    # project_root_main is .../vision_lab/main
    return project_root_main.parent

def create_run_dir(output_root: Path, model: str, run_tag: str = "") -> Path:
    output_root = output_root.resolve()
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    user = getpass.getuser().replace(" ", "_")
    tag = f"__{run_tag}" if run_tag else ""
    run_id = f"{ts}__{model}__user-{user}{tag}__{_rand_suffix()}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # standard subfolders
    for sub in ["checkpoints", "logs", "metrics", "previews", "exports", "mappings"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    return run_dir

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def build_meta(project_root: Path) -> Dict[str, Any]:
    import torch
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
    }

    # optional git commit
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root)).decode().strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = None

    return meta

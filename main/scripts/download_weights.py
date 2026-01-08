# main/scripts/download_weights.py
from __future__ import annotations

import argparse
import sys
from typing import List

from huggingface_hub import HfApi

from fer.inference.hub import (
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    download_to_project,
    where_to_put_weights,
)


def list_repo_model_folders(repo_id: str, revision: str) -> List[str]:
    """
    List top-level folders in the HF repo (these correspond to model folders).
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision=revision)

    folders = set()
    for f in files:
        if "/" in f:
            top = f.split("/", 1)[0].strip()
            if top:
                folders.add(top)

    # filter out obvious non-model entries
    folders.discard(".git")
    return sorted(folders)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download FER weights from Hugging Face into main/src/fer/inference/weights/<model_folder>/"
    )
    ap.add_argument("--repo", default=DEFAULT_REPO_ID, help="HF repo id, e.g. drRamix/EMO_NETS_LMU")
    ap.add_argument("--revision", default=DEFAULT_REVISION, help="HF revision/tag/commit (default: main)")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model folder name, e.g. resnet50, convnext_base, emonext_tiny")
    group.add_argument("--all", action="store_true", help="Download all model folders discovered in the repo")

    ap.add_argument("--force", action="store_true", help="Force re-download even if cached")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would happen")

    args = ap.parse_args()

    if args.all:
        model_folders = list_repo_model_folders(args.repo, args.revision)
        if not model_folders:
            print(f"No folders found in {args.repo}@{args.revision}", file=sys.stderr)
            return 2
        print(f"Found {len(model_folders)} folders in {args.repo}@{args.revision}")
    else:
        model_folders = [args.model]

    ok = 0
    fail = 0

    for mf in model_folders:
        dest = where_to_put_weights(mf)
        if args.dry_run:
            print(f"[DRY] Would download: {args.repo}@{args.revision} -> {dest}")
            continue

        try:
            resolved = download_to_project(
                mf,
                repo_id=args.repo,
                revision=args.revision,
                force=args.force,
            )
            print(f"[OK]   {mf} -> {resolved.folder} ({resolved.format})")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {mf}: {e}", file=sys.stderr)
            fail += 1

    if args.dry_run:
        return 0

    print(f"\nDone. OK={ok}, FAIL={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""
download_sources.py

Downloads / prepares:
- FER2013 (Kaggle competition) via Kaggle CLI:
    kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge

- FERPlus (NO download):
    Builds FERPlus images from FER2013 pixels + local ferplus_labels.csv
    (row i in ferplus_labels.csv corresponds to row i in FER2013 CSV)

- RAF-DB (basic) from:
    URL_RAFDB_ALIGNED_ZIP  (Google Drive file link)
    URL_RAFDB_LABELS       (either Google Drive file link OR local path to a .txt)

Final layout:

vision_lab/main/src/fer/dataset/sources/
  ├── fer2013/
  │    ├── train/{anger,disgust,fear,happiness,sadness,surprise,neutral}/
  │    ├── validation/{...}/
  │    └── test/{...}/
  ├── ferplus/
  │    ├── train/{anger,disgust,fear,happiness,sadness,surprise,neutral}/
  │    ├── validation/{...}/
  │    └── test/{...}/
  └── rafdb/
       ├── train/1..7/
       └── test/1..7/

Additionally mirrors FER2013 into:
vision_lab/main/src/fer/dataset/standardized/fer2013/fer2013_raw/
BUT: inside fer2013_raw, the "neutral" class is removed (train/validation/test).

Auth:
- Kaggle CLI must be installed: pip install kaggle
- Uses Kaggle's single-token auth:
    export KAGGLE_API_TOKEN=KGAT_...
  or put it in .env: KAGGLE_API_TOKEN=KGAT_...
  (script reads it from .env automatically)

Requirements:
- pip install pillow numpy gdown kaggle
- curl must be available
"""

import csv
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ============================================================
# .env helpers
# ============================================================
def _read_env_value(env_path: Path, key: str) -> Optional[str]:
    if not env_path.exists():
        return None
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$")
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.match(line)
        if not m:
            continue
        val = m.group(1).strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1].strip()
        return val or None
    return None


def _get_env_var(key: str) -> Optional[str]:
    v = os.environ.get(key)
    return v.strip() if v and v.strip() else None


def _get_repo_env_paths() -> list[Path]:
    script_path = Path(__file__).resolve()
    project_main = script_path.parents[1]  # .../vision_lab/vision_lab/main
    repo_root = script_path.parents[2]     # .../vision_lab/vision_lab
    outer_root = repo_root.parent          # .../vision_lab
    return [repo_root / ".env", outer_root / ".env", project_main / ".env"]


def _get_required_key(key: str) -> str:
    direct = _get_env_var(key)
    if direct:
        return direct
    for p in _get_repo_env_paths():
        v = _read_env_value(p, key)
        if v:
            print(f"Using {key} from: {p}")
            return v
    tried = "\n".join(str(p) for p in _get_repo_env_paths())
    raise FileNotFoundError(
        f"Missing required key {key}.\n"
        f"Add it to your .env or export it as an env var.\n"
        f"Tried .env paths:\n{tried}"
    )


def _ensure_kaggle_auth() -> None:
    """
    Kaggle supports a single env var:
      KAGGLE_API_TOKEN=KGAT_...
    We'll try to load it from env or from .env automatically.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    try:
        token = _get_required_key("KAGGLE_API_TOKEN")
        os.environ["KAGGLE_API_TOKEN"] = token
        print("Using KAGGLE_API_TOKEN from .env")
    except FileNotFoundError:
        pass


# ============================================================
# Command runner + copy helper
# ============================================================
def _run(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = (p.stdout or "").strip()
    if out:
        print(out)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit {p.returncode}): {' '.join(cmd)}")


def _copytree_overwrite(src: Path, dst: Path) -> None:
    """Copy src -> dst, overwriting dst if it already exists."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


# ============================================================
# FER2013 (download + CSV -> images)
# ============================================================
FER2013_COMPETITION = "challenges-in-representation-learning-facial-expression-recognition-challenge"

IDX_TO_NAME = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}

USAGE_TO_SPLIT = {
    "training": "train",
    "publictest": "validation",
    "privatetest": "test",
}


@dataclass(frozen=True)
class FerRow:
    emotion: int
    usage: str
    pixels: str


def _ensure_class_folders(root: Path) -> None:
    for split in ["train", "validation", "test"]:
        for _, cls_name in IDX_TO_NAME.items():
            (root / split / cls_name).mkdir(parents=True, exist_ok=True)


def _download_kaggle_competition_zip(work_dir: Path) -> Path:
    _ensure_kaggle_auth()
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Download (Kaggle Competition): {FER2013_COMPETITION} ===")
    _run(["kaggle", "competitions", "download", "-c", FER2013_COMPETITION, "-p", str(work_dir)])
    zips = sorted(work_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No .zip downloaded into: {work_dir}")
    return zips[0]


def _unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _find_fer2013_csv(extracted_dir: Path) -> Path:
    p1 = extracted_dir / "icml_face_data.csv"
    if p1.exists():
        return p1
    p2 = extracted_dir / "train.csv"
    if p2.exists():
        return p2
    for p in extracted_dir.rglob("*.csv"):
        if p.name.lower() == "icml_face_data.csv":
            return p
    for p in extracted_dir.rglob("*.csv"):
        if p.name.lower() == "train.csv":
            return p
    raise FileNotFoundError(f"Could not find icml_face_data.csv or train.csv under: {extracted_dir}")


def _read_fer2013_rows(csv_path: Path) -> list[FerRow]:
    rows: list[FerRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Empty CSV: {csv_path}")

        # Kaggle icml_face_data.csv sometimes has leading spaces in header names.
        fieldmap = {c.strip().lower(): c for c in reader.fieldnames}
        if "emotion" not in fieldmap or "usage" not in fieldmap or "pixels" not in fieldmap:
            raise RuntimeError(f"Unexpected columns in {csv_path}: {reader.fieldnames}")

        c_emotion = fieldmap["emotion"]
        c_usage = fieldmap["usage"]
        c_pixels = fieldmap["pixels"]

        for r in reader:
            emotion_raw = (r.get(c_emotion) or "").strip()
            usage_raw = (r.get(c_usage) or "").strip()
            pixels = (r.get(c_pixels) or "").strip()
            if emotion_raw == "" or usage_raw == "" or pixels == "":
                continue
            try:
                emotion = int(emotion_raw)
            except ValueError:
                continue
            rows.append(FerRow(emotion=emotion, usage=usage_raw, pixels=pixels))

    if not rows:
        raise RuntimeError(f"No usable rows parsed from: {csv_path}")
    return rows


def _pixels_to_img48(pixels_str: str) -> Image.Image:
    vals = np.fromstring(pixels_str, sep=" ", dtype=np.uint8)
    if vals.size != 48 * 48:
        raise RuntimeError(f"Unexpected pixels length {vals.size}, expected 2304")
    arr = vals.reshape((48, 48))
    return Image.fromarray(arr, mode="L")


def _write_fer2013_images(rows: list[FerRow], out_root: Path) -> list[int]:
    # Overwrite existing
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    _ensure_class_folders(out_root)

    labels_by_index: list[int] = []
    print(f"\n=== Build FER2013 images -> {out_root} ===")

    wrote = 0
    for i, row in enumerate(rows):
        usage_key = row.usage.strip().lower()
        split = USAGE_TO_SPLIT.get(usage_key)
        labels_by_index.append(row.emotion)

        if split is None or row.emotion not in IDX_TO_NAME:
            continue

        cls_name = IDX_TO_NAME[row.emotion]
        img = _pixels_to_img48(row.pixels)

        fname = f"fer{i:07d}.png"
        dst = out_root / split / cls_name / fname
        img.save(dst, format="PNG")
        wrote += 1

        if wrote % 5000 == 0:
            print(f"  wrote {wrote} images...")

    print(f"Done FER2013: wrote {wrote} images.")
    return labels_by_index


# ============================================================
# FERPlus (local CSV labels + same pixel images, relabeled)
# ============================================================
FERPLUS_COLUMNS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
    "unknown",
    "NF",
]

FERPLUS_TO_FER2013 = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "sadness": 4,
    "surprise": 5,
}


def _find_ferplus_labels_csv(project_main: Path) -> Path:
    repo_root = project_main.parent
    outer_root = repo_root.parent
    candidates = [
        outer_root / "ferplus_labels.csv",
        outer_root / "ferplus-labels.csv",
        repo_root / "ferplus_labels.csv",
        repo_root / "ferplus-labels.csv",
        project_main / "ferplus_labels.csv",
        project_main / "ferplus-labels.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find ferplus_labels.csv (or ferplus-labels.csv).\n"
        "Looked in:\n" + "\n".join(str(p) for p in candidates)
    )


def _read_ferplus_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Empty FERPlus labels file: {csv_path}")
        return list(reader)


def _get_ci(d: dict[str, str], name: str) -> str:
    if name in d:
        return d[name] or ""
    for k in d.keys():
        if k.strip().lower() == name.strip().lower():
            return d[k] or ""
    return ""


def _ferplus_pick_label(ferplus_row: dict[str, str], fer2013_label: int) -> int:
    img_name = _get_ci(ferplus_row, "Image name").strip()
    if img_name == "":
        return fer2013_label

    votes: dict[str, int] = {}
    for col in FERPLUS_COLUMNS:
        raw = _get_ci(ferplus_row, col)
        try:
            votes[col] = int(str(raw).strip()) if str(raw).strip() != "" else 0
        except Exception:
            votes[col] = 0

    best = max(votes.items(), key=lambda kv: kv[1])[0].strip().lower()
    if best in {"neutral", "contempt", "unknown", "nf"}:
        return fer2013_label

    return FERPLUS_TO_FER2013.get(best, fer2013_label)


def _write_ferplus_images(
    fer2013_rows: list[FerRow],
    fer2013_labels_by_index: list[int],
    ferplus_labels_csv: Path,
    out_root: Path,
) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    _ensure_class_folders(out_root)

    print(f"\n=== Build FERPlus images -> {out_root} ===")
    ferplus_rows = _read_ferplus_rows(ferplus_labels_csv)

    n = min(len(fer2013_rows), len(ferplus_rows), len(fer2013_labels_by_index))
    if n == 0:
        raise RuntimeError("FERPlus build: no overlapping rows to process.")

    wrote = 0
    for i in range(n):
        fer_row = fer2013_rows[i]
        fer_label = fer2013_labels_by_index[i]
        fp_row = ferplus_rows[i]

        usage = (_get_ci(fp_row, "Usage") or fer_row.usage).strip()
        split = USAGE_TO_SPLIT.get(usage.lower())
        if split is None:
            continue

        final_label = _ferplus_pick_label(fp_row, fer_label)
        cls_name = IDX_TO_NAME.get(final_label)
        if cls_name is None:
            continue

        img = _pixels_to_img48(fer_row.pixels)
        img_name = _get_ci(fp_row, "Image name").strip() or f"fer{i:07d}.png"

        dst = out_root / split / cls_name / img_name
        img.save(dst, format="PNG")
        wrote += 1

        if wrote % 5000 == 0:
            print(f"  wrote {wrote} images...")

    print(f"Done FERPlus: wrote {wrote} images.")


# ============================================================
# FER2013_raw post-processing: remove "neutral"
# ============================================================
def _remove_neutral_from_fer2013_raw(fer2013_raw_root: Path) -> None:
    """
    Removes the 'neutral' class folder from train/validation/test inside fer2013_raw.
    Expected structure:
      fer2013_raw/{train,validation,test}/neutral/
    """
    removed = 0
    for split in ["train", "validation", "test"]:
        p = fer2013_raw_root / split / "neutral"
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
            removed += 1
    if removed:
        print(f"Removed 'neutral' from fer2013_raw ({removed} split folders).")
    else:
        print("No 'neutral' folders found in fer2013_raw (nothing removed).")


# ============================================================
# RAF-DB (unchanged)
# ============================================================
def _extract_file_id(url: str) -> Optional[str]:
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def _is_html_file(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        head = p.read_bytes()[:512].lstrip()
    except Exception:
        return False
    return head.startswith(b"<!DOCTYPE html") or head.startswith(b"<html") or b"<html" in head[:200].lower()


def _curl_download_simple(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(["curl", "-LfsS", url, "-o", str(out_path)])


def _curl_download_gdrive_confirm(file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cookie = out_path.parent / "gdrive_cookie.txt"

    html_tmp = out_path.parent / "gdrive_tmp.html"
    _run(
        [
            "curl",
            "-L",
            "-c",
            str(cookie),
            f"https://drive.google.com/uc?export=download&id={file_id}",
            "-o",
            str(html_tmp),
            "-sS",
        ]
    )

    txt = html_tmp.read_text(errors="ignore")
    m = re.search(r"confirm=([0-9A-Za-z_]+)", txt)
    if not m:
        _curl_download_simple(f"https://drive.google.com/uc?export=download&id={file_id}", out_path)
        return

    confirm = m.group(1)
    _run(
        [
            "curl",
            "-L",
            "-b",
            str(cookie),
            f"https://drive.google.com/uc?export=download&confirm={confirm}&id={file_id}",
            "-o",
            str(out_path),
            "-sS",
        ]
    )


def _download_drive_or_local(url_or_path: str, out_path: Path, *, allow_fallback: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = Path(url_or_path).expanduser()
    if p.exists() and p.is_file():
        shutil.copy2(p, out_path)
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"Copied local file is empty: {out_path}")
        return

    if not (url_or_path.startswith("http://") or url_or_path.startswith("https://")):
        raise RuntimeError(f"Not a valid URL and not an existing file path: {url_or_path}")

    try:
        _run([sys.executable, "-m", "gdown", "--fuzzy", url_or_path, "-O", str(out_path)])
    except RuntimeError as e:
        if not allow_fallback:
            raise
        print(f"[WARN] gdown failed for {out_path.name}. Trying curl fallback...\n{e}")
        fid = _extract_file_id(url_or_path)
        if not fid:
            raise RuntimeError(f"Could not extract Google Drive file id from url: {url_or_path}")
        _curl_download_simple(f"https://drive.google.com/uc?export=download&id={fid}", out_path)

    if _is_html_file(out_path):
        if not allow_fallback:
            raise RuntimeError(f"Downloaded HTML instead of file for: {out_path} (fallback disabled)")
        fid = _extract_file_id(url_or_path)
        if not fid:
            raise RuntimeError(f"Could not extract Google Drive file id from url: {url_or_path}")
        print(f"[WARN] Received HTML for {out_path.name}. Trying Google Drive confirm-token download...")
        _curl_download_gdrive_confirm(fid, out_path)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {out_path}")


def _verify_zip(zip_path: Path) -> None:
    with zip_path.open("rb") as f:
        sig = f.read(4)
    if sig != b"PK\x03\x04":
        head = zip_path.read_bytes()[:200]
        raise RuntimeError(
            f"Downloaded file is not a valid ZIP (missing PK header): {zip_path}\n"
            f"First bytes: {head!r}"
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP seems corrupted, first bad file: {bad}")
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Bad ZIP file: {zip_path}") from e


def _parse_labels(label_file: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    bad_lines = 0

    with label_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                bad_lines += 1
                continue

            name = parts[0].strip()
            try:
                lab = int(parts[1])
            except Exception:
                bad_lines += 1
                continue

            mapping[name] = lab

    if not mapping:
        preview = label_file.read_text(errors="ignore")[:300]
        raise RuntimeError(
            "Labels file could not be parsed (no valid lines found).\n"
            f"First 300 chars:\n{preview}"
        )

    if bad_lines > 0:
        print(f"[INFO] Labels: skipped {bad_lines} non-conforming lines (ok).")

    return mapping


def _find_aligned_image(aligned_dir: Path, label_name: str) -> Optional[Path]:
    stem = Path(label_name).stem
    for c in [
        aligned_dir / f"{stem}_aligned.jpg",
        aligned_dir / f"{stem}_aligned.png",
        aligned_dir / f"{stem}.jpg",
        aligned_dir / f"{stem}.png",
    ]:
        if c.exists():
            return c
    return None


def _prepare_rafdb_from_drive_files(sources_dir: Path, overwrite: bool = True) -> Path:
    zip_url = _get_required_key("URL_RAFDB_ALIGNED_ZIP")
    labels_src = _get_required_key("URL_RAFDB_LABELS")

    target = sources_dir / "rafdb"
    if overwrite and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    project_main = Path(__file__).resolve().parents[1]
    work_dir = project_main / "scripts" / "_tmp_rafdb_gdrive_files"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    keep_tmp = True
    aligned_zip = work_dir / "aligned.zip"
    labels_txt = work_dir / "list_partition_label.txt"

    try:
        print("\n=== Download (Google Drive / local): RAF-DB basic (direct files) ===")
        print(f"ZIP   : {zip_url}")
        print(f"Labels: {labels_src}")

        _download_drive_or_local(zip_url, aligned_zip, allow_fallback=True)
        _verify_zip(aligned_zip)
        _download_drive_or_local(labels_src, labels_txt, allow_fallback=True)

        if _is_html_file(labels_txt):
            preview = labels_txt.read_text(errors="ignore")[:400]
            raise RuntimeError(
                "Labels download returned HTML instead of text.\n"
                f"First 400 chars:\n{preview}\n"
                "Fix: use a local labels file OR ensure Drive file is publicly accessible."
            )

        extracted = work_dir / "extracted"
        _unzip(aligned_zip, extracted)

        aligned_dir: Optional[Path] = None
        for cand in [
            extracted / "aligned",
            extracted / "Image" / "aligned",
            extracted / "Images" / "aligned",
            extracted / "Image" / "Aligned",
            extracted / "Images" / "Aligned",
        ]:
            if cand.exists():
                aligned_dir = cand
                break
        if aligned_dir is None:
            found = [p for p in extracted.rglob("*") if p.is_dir() and p.name.lower() == "aligned"]
            aligned_dir = found[0] if found else None
        if aligned_dir is None or not aligned_dir.exists():
            raise FileNotFoundError(f"Could not find an 'aligned' directory after unzip at: {extracted}")

        label_map = _parse_labels(labels_txt)

        for split in ["train", "test"]:
            for cls in range(1, 8):
                (target / split / str(cls)).mkdir(parents=True, exist_ok=True)

        copied = 0
        missing = 0
        for label_name, cls in label_map.items():
            if cls < 1 or cls > 7:
                continue
            if label_name.startswith("train_"):
                split = "train"
            elif label_name.startswith("test_"):
                split = "test"
            else:
                continue

            src = _find_aligned_image(aligned_dir, label_name)
            if src is None:
                missing += 1
                continue

            dst = target / split / str(cls) / src.name
            if dst.exists():
                continue
            shutil.copy2(src, dst)
            copied += 1

        (target / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(labels_txt, target / "meta" / labels_txt.name)

        print(f"Placed into       : {target}")
        print(f"Copied images     : {copied}")
        if missing:
            print(f"WARNING missing   : {missing}")

        keep_tmp = False
        return target

    finally:
        if not keep_tmp:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"[INFO] Keeping temp folder for debugging: {work_dir}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    project_main = Path(__file__).resolve().parents[1]  # .../main

    sources_dir = project_main / "src" / "fer" / "dataset" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    standardized_fer2013_dir = project_main / "src" / "fer" / "dataset" / "standardized" / "fer2013"
    standardized_fer2013_dir.mkdir(parents=True, exist_ok=True)

    # ---- FER2013 download + build
    work_dir = project_main / "scripts" / "_tmp_fer2013_kaggle"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    zip_path = _download_kaggle_competition_zip(work_dir)
    extracted_dir = work_dir / "extracted"
    _unzip(zip_path, extracted_dir)

    fer_csv = _find_fer2013_csv(extracted_dir)
    print(f"Using FER2013 CSV: {fer_csv}")

    fer_rows = _read_fer2013_rows(fer_csv)

    fer2013_out = sources_dir / "fer2013"
    fer2013_labels_by_index = _write_fer2013_images(fer_rows, fer2013_out)

    # ---- Mirror FER2013 into standardized/fer2013/fer2013_raw
    fer2013_raw_out = standardized_fer2013_dir / "fer2013_raw"
    _copytree_overwrite(fer2013_out, fer2013_raw_out)
    _remove_neutral_from_fer2013_raw(fer2013_raw_out)  # <-- NEW
    print(f"Mirrored FER2013 into: {fer2013_raw_out}")

    # ---- FERPlus build (from local labels csv + FER2013 pixels)
    ferplus_labels_csv = _find_ferplus_labels_csv(project_main)
    print(f"Using FERPlus labels CSV: {ferplus_labels_csv}")

    ferplus_out = sources_dir / "ferplus"
    _write_ferplus_images(fer_rows, fer2013_labels_by_index, ferplus_labels_csv, ferplus_out)

    # ---- RAF-DB
    _prepare_rafdb_from_drive_files(sources_dir, overwrite=True)

    # cleanup fer tmp
    shutil.rmtree(work_dir, ignore_errors=True)

    print("\nDone. Sources ready:")
    print(f" - {sources_dir/'fer2013'}")
    print(f" - {standardized_fer2013_dir/'fer2013_raw'}  (neutral removed)")
    print(f" - {sources_dir/'ferplus'}")
    print(f" - {sources_dir/'rafdb'}")


if __name__ == "__main__":
    main()

from __future__ import annotations
from pathlib import Path
import sys
import re
import shutil
import urllib.request
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/b/bargozideh/vision_lab/vision_lab/main/src")
from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper

BASE = Path("/home/b/bargozideh/vision_lab/vision_lab/testing")
OUT = BASE / "jaffe"
PNG_ROOT = OUT / "png"
NPY_ROOT = OUT / "npy"
TMP = OUT / "tmp"
ZIP_PATH = TMP / "jaffe.zip"
UNZIPPED = TMP / "unzipped"
URL = "https://zenodo.org/records/14974867/files/jaffe.zip?download=1"

TARGET_SIZE = (64, 64)
MEAN = np.array([0.5368643937226548, 0.5368643937226548, 0.5368643937226548], dtype=np.float32)
STD = np.array([0.21881686437050069, 0.21881686437050069, 0.21881686437050069], dtype=np.float32)

EMO_MAP = {"AN": "anger", "DI": "disgust", "FE": "fear", "HA": "happiness", "SA": "sadness", "SU": "surprise"}

def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)

def unzip(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)

def find_image_root(unzipped: Path) -> Path:
    p = unzipped / "jaffe"
    if p.exists() and p.is_dir():
        return p
    dirs = [d for d in unzipped.iterdir() if d.is_dir()]
    if len(dirs) == 1:
        return dirs[0]
    return unzipped

def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
            yield p

def emo_from_name(name: str):
    stem = Path(name).stem.upper()
    m = re.search(r"\.(AN|DI|FE|HA|SA|SU)\d", stem)
    if not m:
        return None
    return EMO_MAP.get(m.group(1))

def to_gray3(img: Image.Image):
    g = img.convert("L")
    return Image.merge("RGB", (g, g, g))

def to_01(img_rgb: Image.Image):
    return (np.asarray(img_rgb, dtype=np.float32) / 255.0).astype(np.float32)

def z_norm(arr_01: np.ndarray):
    return ((arr_01 - MEAN) / STD).astype(np.float32)

def main():
    for e in EMO_MAP.values():
        (PNG_ROOT / e).mkdir(parents=True, exist_ok=True)
        (NPY_ROOT / e).mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    download(URL, ZIP_PATH)
    unzip(ZIP_PATH, UNZIPPED)

    img_root = find_image_root(UNZIPPED)

    cropper = MTCNNFaceCropper(keep_all=True, min_prob=0.0, width_half=1.3, crop_scale=1.15)

    saved_png = 0
    saved_npy = 0
    skipped = 0
    noface = 0
    total = 0

    for img_path in iter_images(img_root):
        total += 1
        emo = emo_from_name(img_path.name)
        if emo is None:
            skipped += 1
            continue

        img = Image.open(img_path)
        img = to_gray3(img)

        results = cropper.process_pil(img)
        if not results:
            noface += 1
            continue

        crop = to_gray3(results[0].crop).resize(TARGET_SIZE, Image.BILINEAR)

        png_out = PNG_ROOT / emo / f"{img_path.stem}.png"
        if not png_out.exists():
            crop.save(png_out)
            saved_png += 1

        arr_01 = to_01(crop)
        arr_zn = z_norm(arr_01)

        npy_out = NPY_ROOT / emo / f"{img_path.stem}.npy"
        if not npy_out.exists():
            np.save(npy_out, arr_zn)
            saved_npy += 1

    counts_png = {v: len(list((PNG_ROOT / v).glob("*.png"))) for v in EMO_MAP.values()}
    counts_npy = {v: len(list((NPY_ROOT / v).glob("*.npy"))) for v in EMO_MAP.values()}

    print(f"root={img_root}")
    print(f"total={total} saved_png={saved_png} saved_npy={saved_npy} skipped={skipped} noface={noface}")
    print(
        "png: anger={anger} disgust={disgust} fear={fear} happiness={happiness} sadness={sadness} surprise={surprise}".format(
            anger=counts_png["anger"],
            disgust=counts_png["disgust"],
            fear=counts_png["fear"],
            happiness=counts_png["happiness"],
            sadness=counts_png["sadness"],
            surprise=counts_png["surprise"],
        )
    )
    print(
        "npy: anger={anger} disgust={disgust} fear={fear} happiness={happiness} sadness={sadness} surprise={surprise}".format(
            anger=counts_npy["anger"],
            disgust=counts_npy["disgust"],
            fear=counts_npy["fear"],
            happiness=counts_npy["happiness"],
            sadness=counts_npy["sadness"],
            surprise=counts_npy["surprise"],
        )
    )
    print(str(OUT))

if __name__ == "__main__":
    main()




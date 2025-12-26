# setx KAGGLE_API_TOKEN <DEIN API TOKEN>
# pip install kagglehub

"""
download_sources.py

Lädt FER2013 und AffectNet über kagglehub herunter
und kopiert sie nach:

vision_lab/src/fer/dataset/sources/
  ├── fer2013/
  └── affectnet/

Voraussetzungen:
- pip install kagglehub
- kaggle API Key eingerichtet (~/.kaggle/kaggle.json)
"""

from pathlib import Path
import shutil
import kagglehub


# -------------------------------------------------
# Projektpfade
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # vision_lab/
SOURCES_DIR = PROJECT_ROOT / "main" / "src" / "fer" / "dataset" / "sources"

FER2013_DST = SOURCES_DIR / "fer2013"
AFFECTNET_DST = SOURCES_DIR / "affectnet"


# -------------------------------------------------
# Helper: kopieren ohne alles neu zu löschen
# -------------------------------------------------
def copy_dataset(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            # bereits vorhanden → überspringen
            continue

        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


# -------------------------------------------------
# FER2013
# -------------------------------------------------
def download_fer2013():
    print("\n=== Downloading FER2013 ===")
    path = kagglehub.dataset_download("msambare/fer2013")
    src = Path(path)

    print("Kaggle cache path:", src)
    print("Copying to:", FER2013_DST)

    copy_dataset(src, FER2013_DST)
    print("FER2013 ready.")


# -------------------------------------------------
# AffectNet
# -------------------------------------------------
def download_affectnet():
    print("\n=== Downloading AffectNet ===")
    path = kagglehub.dataset_download("mstjebashazida/affectnet")
    src = Path(path)

    print("Kaggle cache path:", src)
    print("Copying to:", AFFECTNET_DST)

    copy_dataset(src, AFFECTNET_DST)
    print("AffectNet ready.")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    download_fer2013()
    download_affectnet()

    print("\nAll datasets downloaded and copied successfully.")
    print("Sources directory:", SOURCES_DIR)


if __name__ == "__main__":
    main()

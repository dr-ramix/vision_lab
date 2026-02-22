# Data Download and Pre-Processing

This document describes the active dataset pipeline implemented by the scripts in `main/scripts`.

## 1. Required Configuration

Create a `.env` file in the repository root (`vision_lab/.env`) and set:

- `KAGGLE_API_TOKEN`: token for FER2013 Kaggle competition download.
- `URL_RAFDB_ALIGNED_ZIP`: RAF-DB aligned images zip URL (Google Drive link).
- `URL_RAFDB_LABELS`: RAF-DB labels source (Google Drive link or local file path).

Reference template: `.env.example`.

## 2. Download Source Datasets

Run:

```bash
cd main/scripts
python download_sources.py
```

What it creates:

- `main/src/fer/dataset/sources/fer2013`
- `main/src/fer/dataset/sources/ferplus`
- `main/src/fer/dataset/sources/rafdb`
- `main/src/fer/dataset/standardized/fer2013/fer2013_raw`

Notes from current implementation:

- FER2013 is downloaded from Kaggle competition files.
- FERPlus is built from FER2013 pixels + `ferplus_labels.csv`.
- In `fer2013_raw`, split name becomes `val` (not `validation`) and `neutral` is removed.

## 3. Build Unified Raw Splits (`images_raw`)

Run:

```bash
cd main/scripts
python prepare_images_raw.py --mode copy --on-conflict overwrite
```

Output:

- `main/src/fer/dataset/standardized/images_raw/{train,val,test}/{anger,disgust,fear,happiness,sadness,surprise}`
- `main/src/fer/dataset/splits/images_raw_manifest.json`

Important behavior:

- Current script merges datasets from `sources/rafdb` and `sources/ferplus`.
- If a dataset has no validation split, validation is carved from train using deterministic hashing.

## 4. Face Crop Pipeline (Mixed + Grey)

Run:

```bash
cd main/scripts
python run_mtcnn_colored_grey.py
```

Output roots:

- `main/src/fer/dataset/standardized/only_mtcnn_cropped/color_and_grey/{png,npy}`
- `main/src/fer/dataset/standardized/only_mtcnn_cropped/grey/{png,npy}`

## 5. Compute Dataset Statistics

Run:

```bash
cd main/scripts
python compute_mean_std.py
```

Creates:

- `.../only_mtcnn_cropped/color_and_grey/dataset_stats_train.json`
- `.../only_mtcnn_cropped/grey/dataset_stats_train.json`

These files are required by the corresponding dataloaders.

## Optional FER2013-Only Branch

If you specifically use FER2013-only standardized data:

```bash
cd main/scripts
python run_mtcnn_crop_fer2013_only.py
python compute_mean_std_fer2013.py
```

Outputs are under:

- `main/src/fer/dataset/standardized/fer2013/fer2013_mtcnn_cropped`

## SLURM Wrappers

Top-level wrappers exist for cluster execution:

- `download_sources.slurm`
- `prepare_images_raw.slurm`

Submit with `sbatch <file>` from repository root.

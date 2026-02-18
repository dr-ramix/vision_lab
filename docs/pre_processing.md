# Pre-processing Pipeline (vision_lab)

This document describes the complete pre-processing pipeline for the `vision_lab` project.  
Follow the steps in the exact order below.

---

## Step 1 — Download all source datasets

First, download all required datasets and metadata by running:

```bash
python vision_lab/main/scripts/download_sources.py
```

This script downloads all raw data sources into the project structure.

---

## Step 2 — Create unified train/val/test structure

After downloading, build the unified dataset structure (`train`, `val`, `test`) by running:

```bash
python vision_lab/main/scripts/prepare_images_raw.py
```

This script:
- Organizes the datasets into a consistent folder structure
- Creates deterministic splits
- Produces the `images_raw` directory used by all subsequent steps

---

# Choose One of the Following Two Variants

After `images_raw` has been created, select exactly **one** of the two processing pipelines.

---

# Variant A — Mixed Dataset  
(FER2013 with FERPlus labels + RAF-DB)

### Step A1 — Face detection and preprocessing (colored + grey)

```bash
python vision_lab/main/scripts/run_mtcnn_colored_grey.py
```

This script:
- Applies MTCNN face detection
- Crops and aligns faces
- Produces the mixed colored/grey dataset

### Step A2 — Compute dataset statistics (mean and std)

```bash
python vision_lab/main/scripts/compute_mean_std.py
```

This script:
- Computes per-channel mean
- Computes per-channel standard deviation
- Outputs normalization statistics used during training

---

# Variant B — FER2013 Only  
(FER2013 without FERPlus labels)

### Step B1 — Face detection and preprocessing

```bash
python vision_lab/main/scripts/run_mtcnn_crop_fer2013_only.py
```

This script:
- Applies MTCNN face detection
- Crops and aligns FER2013 images
- Produces the FER2013-only processed dataset

### Step B2 — Compute dataset statistics (mean and std)

```bash
python vision_lab/main/scripts/compute_mean_std_fer2013.py
```

This script:
- Computes per-channel mean
- Computes per-channel standard deviation
- Outputs normalization statistics for the FER2013-only setup

---

# Full Execution Order

Always run:

1. `python vision_lab/main/scripts/download_sources.py`
2. `python vision_lab/main/scripts/prepare_images_raw.py`

Then choose one:

### Mixed Dataset
3. `python vision_lab/main/scripts/run_mtcnn_colored_grey.py`  
4. `python vision_lab/main/scripts/compute_mean_std.py`

### FER2013 Only
3. `python vision_lab/main/scripts/run_mtcnn_crop_fer2013_only.py`  
4. `python vision_lab/main/scripts/compute_mean_std_fer2013.py`

---

After completing these steps, the dataset is fully pre-processed and ready for model training.

# IMAGE FOLDER CLASSIFICATION – EXECUTION GUIDE

## Script
`submission/classification_model/run_classification.py`

## Purpose
- Takes a folder containing images  
- Detects + crops faces (MTCNN)  
- Applies preprocessing (gray → 3ch → z-norm)  
- Runs FER classification  
- Writes a CSV with class probabilities  

## Available Models
- `vgg19`  
- `resnet18`  
- `coatnetv3_small`  
- `emocatnetsv2_base`  

## Default Mode
Ensemble (**Soft Voting = mean of probabilities**)

## Default Output
`submission/classification_model/classification_scores.csv`

---

## 1️ DEFAULT (Ensemble + Soft Voting)

```bash
python submission/classification_model/run_classification.py \
  --images_dir path/to/images_folder \
  --model ensemble \
  --models vgg19 resnet18 coatnetv3_small emocatnetsv2_base \
  --ensemble mean
```

---

## 2️ Use Single Model

```bash
python submission/classification_model/run_classification.py \
  --images_dir path/to/images_folder \
  --model vgg19
```

---

## 3️ Specify Custom Output CSV

```bash
python submission/classification_model/run_classification.py \
  --images_dir path/to/images_folder \
  --out_csv path/to/output_folder/my_results.csv
```
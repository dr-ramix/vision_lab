# Vision Lab (FER)

This repository contains a **research-oriented PyTorch framework for Facial Expression Recognition (FER)**, developed as part of the **Software Engineering Project (SEP)** in the course **Computer Vision & Deep Learning**.

The goal of this project is to provide a clean, modular, and reproducible framework for:

- Developing and evaluating deep learning architectures for FER  
- Designing structured pre-processing pipelines  
- Running standardized training, evaluation, and inference  
- Enabling image-based classification and video-based FER with explainability (XAI)  

---

## What is Facial Expression Recognition (FER)?

Facial Expression Recognition (FER) is a computer vision task that aims to automatically classify human emotions from facial images or video frames.

Typical emotion classes include:

- anger  
- disgust  
- fear  
- happiness  
- sadness  
- surprise  

FER is used in areas such as:

- Human-computer interaction  
- Affective computing  
- Behavioral analysis  
- Robotics and AI systems  

In this project, we implement and evaluate CNN-based and hybrid deep learning models, structured preprocessing pipelines, and explainability techniques such as Grad-CAM.

---

# Setup Instructions

Please execute the following steps in order.

## 1. Clone the repository

```bash
git clone https://github.com/dr-ramix/vision_lab.git
cd vision_lab
```

## 2. Create and Activate a Virtual Environment

### Linux / macOS

```bash
# create
python -m venv venv
python3 -m venv venv

# activate (bash / sh / zsh)
source venv/bin/activate
. venv/bin/activate

# activate (fish)
source venv/bin/activate.fish

# activate (csh / tcsh)
source venv/bin/activate.csh
```

---

### Windows — PowerShell

```powershell
# create
python -m venv venv
py -m venv venv

# activate (standard)
venv\Scripts\Activate.ps1

# activate (explicit relative path, often required)
.\venv\Scripts\Activate.ps1

# if execution policy blocks activation (temporary fix)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

---

### Windows — CMD

```cmd
:: create
python -m venv venv
py -m venv venv

:: activate
venv\Scripts\activate.bat
```

---

### Windows — Git Bash

```bash
# create
python -m venv venv
py -m venv venv

# activate
source venv/Scripts/activate
```

---

### Deactivate (all platforms)

```bash
deactivate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Install the package

```bash
cd main
pip install -e .
```

This installs the `fer` package in editable mode.

## 5. Verify installation

```bash
python -c "import fer; print('OK')"
```

If successful, it should print:

```
OK
```

---

# Submission Structure

Inside the `submission/` directory you will find:

## 1. Classification Model

Path:

```
submission/classification_model/
```

Documentation:

```
instructions_classification_model.md
```

This explains:

- How to run image-based classification  
- How to perform inference on a folder of images  
- How to use pretrained weights  

---

## 2. Video Demo (FER + XAI)

Path:

```
submission/demo/
```

Documentation:

```
instructions_video_demo.md
```

This explains:

- How to run video-based facial emotion recognition  
- How to apply explainability methods (e.g., Grad-CAM)  
- How to generate predictions with visual explanations  

---


# Data Download & Pre-Processing

After the installation is complete, you must:

- Download the required datasets  
- Configure the `.env` file  
- Run the preprocessing pipeline  

All detailed instructions are documented in:

```
docs/pre_processing.md
```

Follow the steps there carefully and in the specified order.

---

# Project Goal

This repository provides:

- A modular FER research framework  
- Reproducible experiments  
- Extendable model architectures  
- Structured preprocessing  
- Clear separation between training, evaluation, and inference  
- A practical demonstration of explainable AI in FER  


# MM-OVSeg Quick Start Guide

This guide will walk you through the complete setup of the MM-OVSeg repository from scratch, including setting up your Python virtual environment, placing datasets and checkpoints in their correct locations, and running the code.

## 1. Environment Setup (venv)

Since you are on Windows, we will use the built-in `venv` module to create an isolated Python environment. Open a terminal (PowerShell) in the project root directory (`d:\MMOVSeg\MM-OVSeg`) and run:

```powershell
# 1. Create the virtual environment
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\activate

# 3. Install PyTorch with CUDA support (adjust if you need a different version)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install the remaining requirements
pip install -r requirements.txt
```
*(Note: If you plan on using DINOv3 as your backbone, the authors recommend using a newer PyTorch version like `2.5.0` with Python `3.10`).*

---

## 2. Dataset Placement

The codebase uses the `DETECTRON2_DATASETS` environment variable to locate datasets. By default, it expects a directory named `Dataset` inside the project root.

Create a `Dataset` folder at `d:\MMOVSeg\MM-OVSeg\Dataset` and extract your downloaded dataset zip files directly into it so they match the following paths:

*   **OEM (OEM-thin & OEM-thick):** `d:\MMOVSeg\MM-OVSeg\Dataset\OEM`
*   **PIE-RGB-SAR-cloud:** `d:\MMOVSeg\MM-OVSeg\Dataset\PIE_RGBSAR`
*   **DDHR-SK:** `d:\MMOVSeg\MM-OVSeg\Dataset\DDSK`
*   **DDHR-CH:** `d:\MMOVSeg\MM-OVSeg\Dataset\DDCH`

---

## 3. Checkpoint Placement

The model architectures (`MMOV.py` and `MMOVL14.py`) contain hardcoded relative paths that look for a `checkpoint` folder in the root directory.

Create a `checkpoint` folder at `d:\MMOVSeg\MM-OVSeg\checkpoint` and place your downloaded `.pth` weights files exactly as follows:

**For ViT-Base (DINOv1) Models:**
*   `d:\MMOVSeg\MM-OVSeg\checkpoint\RSIB.pth`
*   `d:\MMOVSeg\MM-OVSeg\checkpoint\B16_checkpoint.pth`

**For ViT-Large (DINOv3) Models:**
*   `d:\MMOVSeg\MM-OVSeg\checkpoint\dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
*   `d:\MMOVSeg\MM-OVSeg\checkpoint\L14_checkpoint.pth`

---

## 4. Configuration & Backbone Switching

### A. Set the Dataset Path
Before running the code, set the `DETECTRON2_DATASETS` environment variable so the scripts can find your datasets. In PowerShell, run:
```powershell
$env:DETECTRON2_DATASETS="d:\MMOVSeg\MM-OVSeg\Dataset"
```

### B. Switch the Backbone (Optional)
The project provides a utility script to easily toggle the model backbone between **ViT-B/16 (DINOv1)** and **ViT-L/14 (DINOv3)**. To switch backbones, simply run:
```powershell
python switch_backbone.py
```
*This automatically comments/uncomments the correct imports in `mmov_seg/__init__.py`.*

---

## 5. Running the Code

With the environment activated, variables set, and files in place, you are ready to execute the training and evaluation scripts.

### Training Example (ViT-B/16 on PIEclean):
```powershell
python train_net.py --config configs/vitb_384.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 4 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR output/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "datasets/PIE4class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/PIE.json" DATASETS.TEST '("PIE_valrgb_sem_seg",)' 
```

### Evaluation Example (ViT-B/16 on PIEclean):
```powershell
python train_net.py --eval-only --config configs/vitb_384.yaml --num-gpus 1 --dist-url "auto" MODEL.SEM_SEG_HEAD.NUM_CLASSES 6 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 255 TEST.EVAL_PERIOD 0 OUTPUT_DIR eval/ MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "datasets/PIE4class.json" MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/PIE.json" DATASETS.TRAIN '("PIE_trainrgb_sem_seg",)' DATASETS.TEST '("PIE_valrgb_sem_seg",)' MODEL.WEIGHTS output/B16/modelPIEclean.pth
```
*(Ensure `MODEL.WEIGHTS` points to a valid model that you've trained or downloaded into the `output/` folder).*

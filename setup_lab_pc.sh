#!/bin/bash
# MM-OVSeg Lab PC Setup Script
# Run this script after cloning the repository on your lab PC

echo "=========================================="
echo "    Setting up MM-OVSeg Environment       "
echo "=========================================="

# 1. Create and activate Conda environment
echo "[1/4] Creating Conda environment 'MMOVSeg' with Python 3.8..."
conda create -n MMOVSeg python=3.8 -y

# Initialize conda in this bash script
eval "$(conda shell.bash hook)"
conda activate MMOVSeg

# 2. Install PyTorch (matching the versions in README)
echo "[2/4] Installing PyTorch 2.3.0 with CUDA 11.8..."
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Install Requirements
echo "[3/4] Installing Python requirements..."
pip install -r requirements.txt

# 4. Install Detectron2
echo "[4/4] Installing local detectron2 module..."
python -m pip install -e detectron2

# 5. Download Pretrained Weights
echo "[5/5] Downloading pretrained ViT-B/16 weights..."
python download_weights.py

echo "=========================================="
echo " Setup Complete! "
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment manually: conda activate MMOVSeg"
echo "2. Place your datasets in: $(pwd)/Dataset/"
echo "3. Run your training script: bash trainMMOVSeg.sh (Make sure to run the B16 commands)"
echo ""

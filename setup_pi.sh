#!/bin/bash

# setup_pi.sh - Raspberry Pi 5 Environment Setup for LW-CNN
# Handles PEP 668 by creating a virtual environment.

set -e  # Exit on error

echo "============================================"
echo "   Raspberry Pi 5 Setup for LW-CNN"
echo "============================================"

# 1. System Updates
echo "[1/5] Updating System Packages..."
sudo apt update && sudo apt upgrade -y

# 2. Install System Dependencies
echo "[2/5] Installing System Dependencies..."
# libatlas-base-dev: required for numpy
# libcamera-apps: required for camera
# python3-tk: required for customtkinter/tkinter
# python3-venv: required to create virtual environment
sudo apt install -y libatlas-base-dev libcamera-apps python3-tk python3-venv python3-pip

# 3. Create Virtual Environment (PEP 668 Compliant)
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "[3/5] Virtual Environment '$VENV_DIR' already exists."
else
    echo "[3/5] Creating Virtual Environment '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
fi

# Activate venv for the rest of the script
source $VENV_DIR/bin/activate

# 4. Upgrade Pip in Venv
echo "[4/5] Upgrading Pip..."
pip install --upgrade pip

# 5. Install Python Libraries
echo "[5/5] Installing Python Libraries..."

# TensorFlow / TFLite
# For Pi 5 (Bookworm aarch64), official TensorFlow wheels can be tricky.
# We will try to install tflite-runtime first as it is lighter and sufficient for inference.
# If you need full training, you might need a community build.
echo "    - Installing tflite-runtime (preferred for inference)..."
# Using a specific extra-index-url for Pi wheels if needed, or just standard pip.
# Google's coral repo often has good tflite runtime, but let's try standard pip first.
pip install tflite-runtime || pip install tensorflow-cpu

# CustomTkinter & UI
echo "    - Installing CustomTkinter & Pillow..."
pip install customtkinter pillow

# OpenCV (Headless to avoid apt dependency hell)
echo "    - Installing OpenCV Headless..."
pip install opencv-python-headless

# Numpy (ensure compatibility)
echo "    - Installing Numpy..."
pip install "numpy<2.0"  # TF often dislikes numpy 2.0+

echo "============================================"
echo "   Setup Complete!"
echo "============================================"
echo "To run your app:"
echo "  source venv/bin/activate"
echo "  python parasite_classifier_app.py"
echo "============================================"

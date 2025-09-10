#!/bin/bash
# ========================================
# Raspberry Pi 5 ML Environment Setup
# For EfficientNet_B0_LW_CNN Notebook
# ========================================

echo "🔄 Updating system..."
sudo apt update && sudo apt upgrade -y

echo "📦 Installing Python & pip..."
sudo apt install -y python3 python3-pip python3-venv

echo "📦 Creating virtual environment..."
python3 -m venv ~/ml-env
source ~/ml-env/bin/activate

echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "📦 Installing Jupyter..."
pip install notebook jupyterlab

echo "📦 Installing core Python libraries..."
pip install numpy pandas matplotlib scikit-learn opencv-python

echo "📦 Installing TensorFlow (ARM build for Raspberry Pi)..."
pip install tensorflow-aarch64

echo "📦 Installing Keras & EfficientNet..."
pip install keras efficientnet

echo "📦 Installing TensorFlow Addons (if used)..."
pip install tensorflow-addons

echo "✅ Setup complete!"
echo "👉 To activate your environment later, run:"
echo "   source ~/ml-env/bin/activate"
echo "👉 To launch Jupyter Notebook, run:"
echo "   jupyter notebook"

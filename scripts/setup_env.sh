#!/bin/bash

# Define environment directory
ENV_DIR="venv"

echo "🔨 Setting up Virtual Environment..."

# 1. Install pip and venv if validation fails
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

# 2. Create Venv
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment in $ENV_DIR..."
    python3 -m venv $ENV_DIR --system-site-packages
else
    echo "✅ Virtual environment already exists."
fi

# 3. Activate and Install
source $ENV_DIR/bin/activate

echo "📦 Installing Requirements..."
# Upgrade pip inside venv
pip install --upgrade pip

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found. Installing defaults..."
    pip install "numpy<2" customtkinter tensorflow opencv-python-headless pillow
fi

echo "✅ Environment Setup Complete."
echo "   To use manually: source $ENV_DIR/bin/activate"

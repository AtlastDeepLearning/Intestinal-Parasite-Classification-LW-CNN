#!/bin/bash

# Define environment directory
ENV_DIR="venv"

echo "🔨 Setting up Virtual Environment (Debian 12+ Compatible)..."

# 1. Ensure python3-venv is installed (Uses apt, so it's allowed)
if ! dpkg -s python3-venv &> /dev/null; then
    echo "📦 Installing python3-venv..."
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-pip
fi

# 2. Create Venv
# We use --system-site-packages so we can access apt-installed packages like python3-opencv
if [ ! -d "$ENV_DIR" ]; then
    echo "creating virtual environment in $ENV_DIR..."
    python3 -m venv $ENV_DIR --system-site-packages
else
    echo "✅ Virtual environment already exists."
fi

# 3. Activate and Install
source $ENV_DIR/bin/activate

echo "📦 Installing Requirements in venv..."

# IMPORTANT: On Raspberry Pi Bookworm, we might need to rely on apt for heavy packages
# but inside a venv, pip is allowed.

# Upgrade pip inside venv
pip install --upgrade pip

# Install dependencies
# We use 'customtkinter' and 'tensorflow' which are best from pip
# We exclude system packages that are better installed via apt (opencv, pilots) if they are in requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found. Installing defaults..."
    pip install "numpy<2" customtkinter tensorflow
fi

echo "✅ Environment Setup Complete."
echo "   To use manually: source $ENV_DIR/bin/activate"

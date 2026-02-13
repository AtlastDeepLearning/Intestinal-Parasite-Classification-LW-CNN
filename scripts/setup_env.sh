#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

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
# Check for broken venv (dir exists but no python binary)
if [ -d "$ENV_DIR" ] && [ ! -f "$ENV_DIR/bin/python3" ]; then
    echo "⚠️  Found broken virtual environment (missing binaries). Recreating..."
    rm -rf "$ENV_DIR"
fi

# We use --system-site-packages so we can access apt-installed packages like python3-opencv
if [ ! -d "$ENV_DIR" ]; then
    echo "creating virtual environment in $ENV_DIR..."
    python3 -m venv $ENV_DIR --system-site-packages
else
    echo "✅ Virtual environment already exists."
fi

# 3. Install Requirements using EXPLICIT venv pip
# We avoid 'source activate' in scripts as it triggers 'externally-managed' easier
VENV_PIP="$ENV_DIR/bin/pip"

echo "📦 Installing Requirements using $VENV_PIP..."

# Upgrade pip inside venv (with break-system-packages just in case)
"$VENV_PIP" install --upgrade pip --break-system-packages

# Install dependencies
if [ -f "requirements.txt" ]; then
    "$VENV_PIP" install -r requirements.txt --break-system-packages
else
    echo "⚠️ requirements.txt not found. Installing defaults..."
    "$VENV_PIP" install "numpy<2" customtkinter tensorflow --break-system-packages
fi

echo "✅ Environment Setup Complete."
echo "   To use manually: source $ENV_DIR/bin/activate"

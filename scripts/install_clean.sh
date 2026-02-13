#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")
# Go to the project root (one level up)
APP_DIR=$(dirname "$SCRIPT_DIR")
cd "$APP_DIR"

echo "🧹 Starting Clean Installation for Raspberry Pi 5 (Bookworm)..."

# 1. Clean old venv to ensure no corruption
if [ -d "venv" ]; then
    echo "🗑️  Removing old virtual environment..."
    rm -rf venv
fi

# 2. System Dependencies (Apt)
# We use apt for OpenCV and heavy math libs because they are pre-compiled for Pi
echo "📦 Installing System Dependencies (this might take a minute)..."
sudo apt-get update
# python3-opencv: Optimized OpenCV for Pi
# libhdf5-dev: Required for H5py (used by Keras)
# libatlas-base-dev: Required for Numpy
sudo apt-get install -y python3-opencv libhdf5-dev libatlas-base-dev python3-venv

# 3. Create Venv (System Site Packages for OpenCV)
# --system-site-packages allows us to use the apt-installed python3-opencv inside the venv
echo "creationg venv with system packages..."
python3 -m venv venv --system-site-packages

# 4. Install Python Libs
echo "📦 Installing Python Libraries..."
# We use --break-system-packages because we are inside a venv (even if it has system packages)
# and we want to ensure pip doesn't block us strictly.
venv/bin/pip install --upgrade pip --break-system-packages
venv/bin/pip install -r requirements.txt --break-system-packages

# 5. Make launcher executable
chmod +x run_app.sh
chmod +x scripts/debug_and_push.sh

echo "---------------------------------------------------"
echo "✅ Clean Install Complete!"
echo "   To start the app: ./run_app.sh"
echo "---------------------------------------------------"

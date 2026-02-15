#!/bin/bash
# ========================================
# Raspberry Pi 5 ML Environment Setup
# EfficientNet_B0_LW_CNN Notebook
# ========================================

set -e  # stop if any command fails

echo "🔄 Updating system..."
sudo apt update && sudo apt upgrade -y

echo "📦 Installing dependencies..."
# Added python3-picamera2 and python3-libcamera
sudo apt install -y python3 python3-pip python3-venv python3-full python3-libcamera python3-picamera2

# Create venv if it doesn’t exist
if [ ! -d "$HOME/ml-env" ]; then
    echo "📦 Creating virtual environment at ~/ml-env (with system-site-packages)"
    # --system-site-packages is required to see apt-installed python3-picamera2
    python3 -m venv --system-site-packages ~/ml-env
fi

echo "📦 Activating virtual environment..."
source ~/ml-env/bin/activate

echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Note: opencv-python might duplicate system opencv, but we keep it for now.
echo "📦 Installing core libraries..."
pip install numpy pandas matplotlib scikit-learn opencv-python

echo "📦 Installing ML frameworks..."
pip install tensorflow-aarch64 keras efficientnet tensorflow-addons

echo "📦 Installing Jupyter Notebook/Lab..."
pip install notebook jupyterlab

echo "🔍 Running environment self-check..."
python3 - <<'EOF'
import sys
try:
    import numpy, pandas, matplotlib, sklearn, cv2
    print("✅ Core libraries installed")
except Exception as e:
    print("❌ Core libraries issue:", e); sys.exit(1)

try:
    import tensorflow as tf
    print("✅ TensorFlow version:", tf.__version__)
except Exception as e:
    print("❌ TensorFlow issue:", e); sys.exit(1)

try:
    import picamera2
    print("✅ Picamera2 installed")
except ImportError:
    print("⚠️ Picamera2 NOT found (Camera features will be disabled)")
except Exception as e:
    print("❌ Picamera2 issue:", e)

print("🎉 Environment setup complete! Ready to run your notebook.")
EOF

echo "👉 To activate your environment later, run:"
echo "   source ~/ml-env/bin/activate"
echo "👉 To launch the App:"
echo "   ./scripts/pi_run_app.sh"

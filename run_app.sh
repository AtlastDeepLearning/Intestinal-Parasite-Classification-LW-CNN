#!/bin/bash

# Get directory of this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
APP_DIR=$(dirname "$SCRIPT_DIR")
VENV_DIR="$APP_DIR/venv"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Running system setup..."
    bash "$SCRIPT_DIR/scripts/install_clean.sh"
fi

# Run the app using the venv python
echo "🚀 Launching Parasite Classifier..."
"$VENV_DIR/bin/python3" "$APP_DIR/parasite_classifier_app.py"

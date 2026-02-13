#!/bin/bash

# Configuration
APP_NAME="ParasiteClassifier"
APP_DIR=$(pwd)
VENV_PYTHON="$APP_DIR/venv/bin/python3"
EXEC_CMD="$VENV_PYTHON $APP_DIR/parasite_classifier_app.py"
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/$APP_NAME.desktop"

# Ensure venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found. Running setup_env.sh..."
    bash scripts/setup_env.sh
fi

# Ensure autostart directory exists
mkdir -p "$AUTOSTART_DIR"

# Create .desktop file
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=Auto-launch Parasite Classifier
Exec=lxterminal -e "bash -c 'cd $APP_DIR && $EXEC_CMD; read -p \"App closed. Press enter to exit...\"'"
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

echo "✅ Autostart entry created at: $DESKTOP_FILE"
echo "The app should now launch automatically on reboot."

#!/bin/bash

# Configuration
APP_NAME="ParasiteClassifier"
APP_DIR=$(pwd)
EXEC_CMD="python3 $APP_DIR/parasite_classifier_app.py"
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/$APP_NAME.desktop"

# Ensure autostart directory exists
mkdir -p "$AUTOSTART_DIR"

# Install dependencies
echo "Installing System Dependencies..."
sudo apt-get update
sudo apt-get install -y python3-opencv python3-pil.imagetk python3-numpy \
    libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libcamera

echo "Installing Python Libraries..."
pip3 install customtkinter tensorflow --break-system-packages

# Create .desktop file
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=Auto-launch Parasite Classifier
Exec=lxterminal -e "bash -c 'cd $APP_DIR && $EXEC_CMD; read -p \"Press enter to close...\"'"
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

echo "✅ Autostart entry created at: $DESKTOP_FILE"
echo "The app should now launch automatically on reboot."

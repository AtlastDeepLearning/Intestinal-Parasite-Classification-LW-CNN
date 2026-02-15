#!/bin/bash

# Define the autostart directory
AUTOSTART_DIR="$HOME/.config/autostart"

# Ensure the directory exists
mkdir -p "$AUTOSTART_DIR"

echo "🧹 Cleaning up old autostart entries..."
# Remove any existing .desktop files that might conflict
rm -f "$AUTOSTART_DIR"/ParasiteClassifier*.desktop
rm -f "$AUTOSTART_DIR"/thesis*.desktop
rm -f "$AUTOSTART_DIR"/lw-cnn*.desktop

echo "✅ Old entries removed."

echo "🚀 Creating new autostart entry..."

# Define the path to the run script (assuming standard install location)
RUN_SCRIPT="$HOME/thesis/scripts/pi_run_app.sh"

# Make sure the run script is executable
chmod +x "$RUN_SCRIPT"

# Create the .desktop file
cat <<EOF > "$AUTOSTART_DIR/parasite-classifier.desktop"
[Desktop Entry]
Type=Application
Name=Parasite Classifier
Exec=$RUN_SCRIPT
Path=$HOME/thesis/thesis
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

echo "🎉 Autostart configured! The app will launch on boot."
echo "   File created at: $AUTOSTART_DIR/parasite-classifier.desktop"

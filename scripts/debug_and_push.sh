#!/bin/bash

LOG_FILE="debug_log_$(date +%Y%m%d_%H%M%S).txt"
APP_DIR=$(pwd)
VENV_PYTHON="$APP_DIR/venv/bin/python3"

echo "🐞 Starting Debug Run..."
echo "----------------------------------------" > "$LOG_FILE"
echo "TS: $(date)" >> "$LOG_FILE"
echo "CWD: $APP_DIR" >> "$LOG_FILE"
echo "PYTHON: $VENV_PYTHON" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"

# 1. Check if venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found at $VENV_PYTHON" | tee -a "$LOG_FILE"
    echo "   Running install_clean.sh..." | tee -a "$LOG_FILE"
    bash scripts/install_clean.sh >> "$LOG_FILE" 2>&1
fi

# 2. Run the App and capture ALL output
echo "🚀 Running App..." | tee -a "$LOG_FILE"
# We use stdbuf to unbuffer output so we see it immediately
if command -v stdbuf &> /dev/null; then
    stdbuf -oL -eL $VENV_PYTHON parasite_classifier_app.py >> "$LOG_FILE" 2>&1
else
    $VENV_PYTHON parasite_classifier_app.py >> "$LOG_FILE" 2>&1
fi
EXIT_CODE=$?

echo "----------------------------------------" >> "$LOG_FILE"
echo "App exited with code: $EXIT_CODE" >> "$LOG_FILE"

# 3. Git Push the Log
echo "📤 Pushing log to GitHub..."
git pull origin main --no-rebase # Ensure we are up to date
git add "$LOG_FILE"
git commit -m "Add debug log from Pi: $LOG_FILE"
git push origin main

echo "✅ Done! Check GitHub for $LOG_FILE"

# 🍓 Raspberry Pi 5 Deployment Guide

This guide explains how to set up and run the Parasite Classifier on a Raspberry Pi 5 with the **Raspberry Pi Camera Module 3 or HQ Camera**.

## 📂 File Manifest

Ensure these files are present on your Pi (e.g., in `~/Downloads/thesis/`):

- `main_pi_controller.py`: The main GUI application.
- `models/efficientnet_b0_parasite.h5`: The trained AI model.
- `scripts/pi_env_setup.sh`: Installation script.
- `scripts/pi_run_app.sh`: Launch script.

---

## 🛠️ Step 1: Installation

1.  Open a terminal on your Raspberry Pi.
2.  Navigate to your project folder:
    ```bash
    cd ~/Downloads/thesis
    ```
3.  Make the scripts executable:
    ```bash
    chmod +x scripts/*.sh
    ```
4.  Run the setup script (internet required):
    ```bash
    ./scripts/pi_env_setup.sh
    ```
    > **Note:** This will install system dependencies (including `python3-picamera2`), create a virtual environment, and install TensorFlow. It may take 10-20 minutes.

---

## 🚀 Step 2: Running the App

Once installation is complete, launch the application:

```bash
./scripts/pi_run_app.sh
```

---

## 📸 Helper Guide: Using the App

1.  **Start Camera**: Click **"🎥 Capture (Start)"** to see the live video feed.
2.  **Aim & Focus**: Adjust your microscope/camera until the parasite is visible.
3.  **Snap**: Click **"📸 Snap (Freeze)"** to freeze the image.
4.  **Crop**:
    - Click on the image to set the top-left corner of the crop box.
    - The box is fixed at **224x224 pixels** (optimal for the AI).
    - **Drag** the mouse to position the red box around the parasite.
5.  **Classify**: Click **"🔬 Classify"** to analyze the content of the red box.
6.  **Results**: The top 3 predictions will appear in the window.

### ❓ Troubleshooting

- **"Picamera2 not found"**: Ensure you ran the updated `pi_env_setup.sh` which installs `python3-picamera2`.
- **Camera not working**: Run `libcamera-hello` in a terminal to verify the hardware connection.
- **Model error**: Ensure `models/efficientnet_b0_parasite.h5` exists.

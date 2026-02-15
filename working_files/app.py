import tensorflow as tf
import numpy as np
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# ---------------------------
# Config
# ---------------------------

MODEL_PATH = "efficientnet_parasite_final.h5"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "ascaris_lumbricoides",
    "enterobius_vermicularis",
    "hookworm",
    "trichuris_trichiura",
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------
# Model Definition + Loading
# ---------------------------

def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    base = EfficientNetB0(include_top=False, weights=None, input_shape=input_shape, pooling="avg")
    outputs = layers.Dense(num_classes, activation="softmax")(base.output)
    model = models.Model(inputs=base.input, outputs=outputs)
    return model

def load_model(model_path: str):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        if "Shape mismatch" in str(e) or "expects shape" in str(e):
            model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES)
            model.load_weights(model_path)
            return model
        raise

# ---------------------------
# Image Preprocessing
# ---------------------------

def preprocess_image_inference(image_path: str) -> np.ndarray:
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=IMG_SIZE, color_mode="grayscale"
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # (H, W, 1)
    img_array = np.repeat(img_array, 3, axis=-1)                # (H, W, 3)
    img_array = preprocess_input(img_array)                     # EfficientNet preprocessing
    img_array = np.expand_dims(img_array, axis=0)               # (1, H, W, 3)
    return img_array

# ---------------------------
# Prediction
# ---------------------------

def predict_image(model, image_path: str, top_k: int = 3):
    batch = preprocess_image_inference(image_path)
    preds = model.predict(batch, verbose=0)[0]  # shape (NUM_CLASSES,)
    top_indices = preds.argsort()[-top_k:][::-1]
    results = [(CLASS_NAMES[i], float(preds[i])) for i in top_indices]
    return results

# ---------------------------
# GUI
# ---------------------------

_gui_model = None
_gui_lock = threading.Lock()

def _ensure_model_loaded():
    global _gui_model
    with _gui_lock:
        if _gui_model is None:
            _gui_model = load_model(MODEL_PATH)
    return _gui_model

def launch_gui():
    root = tk.Tk()
    root.title("Parasite Classifier")
    root.geometry("600x500")

    selected_path_var = tk.StringVar(value="No image selected")

    # Preview area
    preview_label = tk.Label(root, text="Preview", width=50, height=15, relief=tk.SUNKEN)
    preview_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Results area
    results_var = tk.StringVar(value="Top predictions will appear here")
    results_label = tk.Label(root, textvariable=results_var, justify=tk.LEFT, anchor="w")
    results_label.pack(padx=10, pady=(0, 10), fill=tk.X)

    # Path label
    path_label = tk.Label(root, textvariable=selected_path_var, anchor="w")
    path_label.pack(padx=10, pady=(0, 10), fill=tk.X)

    def choose_image():
        try:
            path = filedialog.askopenfilename(
                title="Choose an image",
                filetypes=[
                    ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return

            selected_path_var.set(path)

            # Show thumbnail preview
            img = Image.open(path)
            img.thumbnail((400, 400))
            tk_img = ImageTk.PhotoImage(img)
            preview_label.configure(image=tk_img)
            preview_label.image = tk_img  # keep reference

            # Run prediction in background to keep UI responsive
            def run_prediction():
                try:
                    results_var.set("Loading model and predicting...")
                    model = _ensure_model_loaded()
                    results = predict_image(model, path, top_k=3)
                    formatted = "\n".join([f"{name}: {score:.4f}" for name, score in results])
                    results_var.set(formatted)
                except Exception:
                    traceback.print_exc()
                    messagebox.showerror("Error", "Failed to predict. See console for details.")

            threading.Thread(target=run_prediction, daemon=True).start()

        except Exception as ex:
            traceback.print_exc()
            messagebox.showerror("Error", str(ex))

    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(padx=10, pady=10, fill=tk.X)

    choose_btn = tk.Button(btn_frame, text="Choose Image", command=choose_image)
    choose_btn.pack(side=tk.LEFT)

    quit_btn = tk.Button(btn_frame, text="Quit", command=root.destroy)
    quit_btn.pack(side=tk.RIGHT)

    root.mainloop()

# ---------------------------
# CLI Main (optional quick test)
# ---------------------------

if __name__ == "__main__":
    # Launch the GUI by default. You can still run CLI quick test by
    # commenting the next line and uncommenting the CLI block below.
    launch_gui()

    # CLI quick test example (disabled by default):
    # image_path = "C:/Users/Atlast/Downloads/testing/testing/ascaris.png"
    # model = load_model(MODEL_PATH)
    # results = predict_image(model, image_path, top_k=3)
    # print("Top predictions:")
    # for name, score in results:
    #     print(f"  {name}: {score:.4f}")

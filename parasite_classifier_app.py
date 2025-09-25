import tensorflow as tf
import numpy as np
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
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
# Model Handling
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
# Image Preprocessing + Prediction
# ---------------------------
def preprocess_image_inference(image_path: str) -> np.ndarray:
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=IMG_SIZE, color_mode="grayscale"
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, image_path: str, top_k: int = 3):
    batch = preprocess_image_inference(image_path)
    preds = model.predict(batch, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    results = [(CLASS_NAMES[i], float(preds[i])) for i in top_indices]
    return results


# ---------------------------
# GUI with Camera + Touch Controls
# ---------------------------
_gui_model = None
_gui_lock = threading.Lock()


def _ensure_model_loaded():
    global _gui_model
    with _gui_lock:
        if _gui_model is None:
            _gui_model = load_model(MODEL_PATH)
    return _gui_model


class ParasiteApp:
    def __init__(self, root):
        self.root = root
        root.title("Parasite Classifier")
        root.geometry("800x600")

        self.selected_path = None
        self.captured_image = None
        self.crop_rect = None
        self.crop_start = None

        self.results_var = tk.StringVar(value="Top predictions will appear here")

        # Preview area
        self.preview_label = tk.Label(root, text="Preview", width=80, height=25, relief=tk.SUNKEN, bg="black")
        self.preview_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Results
        results_label = tk.Label(root, textvariable=self.results_var, justify=tk.LEFT, anchor="w", font=("Arial", 14))
        results_label.pack(padx=10, pady=(0, 10), fill=tk.X)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(padx=10, pady=10, fill=tk.X)

        tk.Button(btn_frame, text="Open Camera", command=self.open_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Capture", command=self.capture_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Crop", command=self.enable_crop).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Predict", command=self.run_prediction).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Quit", command=root.destroy).pack(side=tk.RIGHT, padx=5)

        # Crop bindings
        self.preview_label.bind("<ButtonPress-1>", self.start_crop)
        self.preview_label.bind("<B1-Motion>", self.draw_crop)
        self.preview_label.bind("<ButtonRelease-1>", self.finish_crop)

        self.cap = None  # OpenCV video capture

    def open_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.show_frame()

    def show_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.captured_image = frame
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                img.thumbnail((600, 400))
                imgtk = ImageTk.PhotoImage(image=img)
                self.preview_label.imgtk = imgtk
                self.preview_label.configure(image=imgtk)
            self.root.after(30, self.show_frame)

    def capture_image(self):
        if self.captured_image is not None:
            cv2.imwrite("captured_image.jpg", self.captured_image)
            self.selected_path = "captured_image.jpg"
            self.results_var.set("Image captured, ready for crop/predict")

    def enable_crop(self):
        self.results_var.set("Draw rectangle on image to crop")

    def start_crop(self, event):
        if self.selected_path:
            self.crop_start = (event.x, event.y)
            self.crop_rect = None

    def draw_crop(self, event):
        if self.crop_start:
            self.crop_rect = (self.crop_start[0], self.crop_start[1], event.x, event.y)
            # Draw rectangle overlay (not permanent)
            img = Image.open(self.selected_path)
            img.thumbnail((600, 400))
            overlay = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=overlay)
            self.preview_label.image = overlay

    def finish_crop(self, event):
        if self.crop_rect and self.selected_path:
            x1, y1, x2, y2 = self.crop_rect
            img = Image.open(self.selected_path)
            img.thumbnail((600, 400))
            cropped = img.crop((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            cropped.save("cropped_image.jpg")
            self.selected_path = "cropped_image.jpg"

            tk_img = ImageTk.PhotoImage(cropped)
            self.preview_label.configure(image=tk_img)
            self.preview_label.image = tk_img
            self.results_var.set("Cropped image ready")

    def run_prediction(self):
        if not self.selected_path:
            messagebox.showwarning("No Image", "Please capture or choose an image first.")
            return

        def task():
            try:
                self.results_var.set("Loading model and predicting...")
                model = _ensure_model_loaded()
                results = predict_image(model, self.selected_path, top_k=3)
                formatted = "\n".join([f"{name}: {score:.4f}" for name, score in results])
                self.results_var.set(formatted)
            except Exception:
                traceback.print_exc()
                messagebox.showerror("Error", "Failed to predict. See console for details.")

        threading.Thread(target=task, daemon=True).start()


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ParasiteApp(root)
    root.mainloop()

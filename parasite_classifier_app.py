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

# 🔹 Add Picamera2
from picamera2 import Picamera2

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
def preprocess_cv_image(cv_img: np.ndarray) -> np.ndarray:
    """ Preprocess cropped OpenCV image for EfficientNet """
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_cv_image(model, cv_img: np.ndarray, top_k: int = 3):
    batch = preprocess_cv_image(cv_img)
    preds = model.predict(batch, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    results = [(CLASS_NAMES[i], float(preds[i])) for i in top_indices]
    return results

# ---------------------------
# Tkinter App
# ---------------------------
class CropperApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.current_image = None   # OpenCV image (BGR)
        self.tk_img_ref = None
        self.rect_id = None
        self.start_x, self.start_y = None, None
        self.crop_box_size = 224    # fixed 224x224 crop box
        self.cropped_image = None

        # Canvas
        self.canvas = tk.Canvas(root, bg="lightgray", width=500, height=400)
        self.canvas.pack(pady=10, expand=True)

        self.canvas.bind("<Button-1>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.update_crop)
        self.canvas.bind("<ButtonRelease-1>", self.finish_crop)

        # Results label
        self.results_var = tk.StringVar(value="Results will appear here")
        tk.Label(root, textvariable=self.results_var, anchor="w", justify="left").pack(fill="x", pady=5)

        # Buttons
        btn_frame = tk.Frame(root, bg="white", height=60)
        btn_frame.pack(side="bottom", fill="x")

        open_btn = tk.Button(btn_frame, text="📂 Open", command=self.open_image, height=2, width=12)
        open_btn.pack(side="left", expand=True, padx=5, pady=10)

        capture_btn = tk.Button(btn_frame, text="📸 Capture", command=self.capture_image, height=2, width=12)
        capture_btn.pack(side="left", expand=True, padx=5, pady=10)

        classify_btn = tk.Button(btn_frame, text="🤖 Classify", command=self.classify_image, height=2, width=12)
        classify_btn.pack(side="right", expand=True, padx=5, pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.show_image(self.current_image)

    def capture_image(self):
        """Capture image using Raspberry Pi HQ Camera"""
        try:
            picam2 = Picamera2()
            config = picam2.create_still_configuration(main={"size": (1920, 1080)})
            picam2.configure(config)
            picam2.start()
            frame = picam2.capture_array()
            picam2.close()

            # Convert RGB (from camera) -> BGR (for OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.current_image = frame_bgr
            self.show_image(self.current_image)

        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {e}")

    def show_image(self, cv_img):
        """Convert OpenCV image to Tkinter Canvas"""
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        img = img.resize((500, 400))
        self.tk_img_ref = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img_ref)

    def start_crop(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def update_crop(self, event):
        if self.start_x and self.start_y:
            if self.rect_id:
                self.canvas.delete(self.rect_id)

            # Force crop to 224x224 box
            x1, y1 = self.start_x, self.start_y
            x2, y2 = x1 + self.crop_box_size, y1 + self.crop_box_size
            self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def finish_crop(self, event):
        if self.current_image is None:
            return

        # Map canvas coords back to image coords
        h, w, _ = self.current_image.shape
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        x1, y1 = self.start_x, self.start_y
        x2, y2 = x1 + self.crop_box_size, y1 + self.crop_box_size

        scale_x = w / canvas_w
        scale_y = h / canvas_h

        ix1, iy1 = int(x1 * scale_x), int(y1 * scale_y)
        ix2, iy2 = int(x2 * scale_x), int(y2 * scale_y)

        self.cropped_image = self.current_image[iy1:iy2, ix1:ix2]
        if self.cropped_image.size > 0:
            self.show_image(self.cropped_image)

    def classify_image(self):
        if self.cropped_image is None:
            messagebox.showwarning("Warning", "No cropped image available!")
            return

        try:
            results = predict_cv_image(self.model, self.cropped_image, top_k=3)
            formatted = "\n".join([f"{name}: {score:.4f}" for name, score in results])
            self.results_var.set(formatted)
        except Exception:
            traceback.print_exc()
            messagebox.showerror("Error", "Classification failed. See console for details.")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    model = load_model(MODEL_PATH)

    root = tk.Tk()
    root.title("Parasite Classifier")
    root.geometry("800x480")  # Fits Pi touchscreen

    app = CropperApp(root, model)

    root.mainloop()

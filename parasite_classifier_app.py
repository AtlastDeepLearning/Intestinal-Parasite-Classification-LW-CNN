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
# Cropper + Classifier GUI
# ---------------------------
class CropperApp:
    def __init__(self, root, model, img_path):
        self.root = root
        self.model = model
        self.root.title("Parasite Classifier - Crop 224x224")

        # Load image
        self.image = Image.open(img_path)
        self.tk_img = ImageTk.PhotoImage(self.image)

        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height, cursor="cross")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # Fixed crop size
        self.crop_w, self.crop_h = 224, 224

        # Start position (centered)
        self.start_x, self.start_y = self.image.width//2 - 112, self.image.height//2 - 112

        # Draw fixed rectangle
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y,
            self.start_x + self.crop_w, self.start_y + self.crop_h,
            outline="red", width=2
        )

        # Dragging
        self.canvas.bind("<B1-Motion>", self.move_crop_box)

        # Crop + classify button
        self.btn = tk.Button(root, text="✂️ Crop & Classify", command=self.save_and_classify)
        self.btn.pack(fill="x")

    def move_crop_box(self, event):
        x1 = event.x - self.crop_w//2
        y1 = event.y - self.crop_h//2
        x2 = x1 + self.crop_w
        y2 = y1 + self.crop_h

        # Clamp inside image
        if x1 < 0: x1, x2 = 0, self.crop_w
        if y1 < 0: y1, y2 = 0, self.crop_h
        if x2 > self.image.width: x1, x2 = self.image.width - self.crop_w, self.image.width
        if y2 > self.image.height: y1, y2 = self.image.height - self.crop_h, self.image.height

        self.start_x, self.start_y = x1, y1
        self.canvas.coords(self.rect, x1, y1, x2, y2)

    def save_and_classify(self):
        cropped = self.image.crop((self.start_x, self.start_y, self.start_x+self.crop_w, self.start_y+self.crop_h))
        cropped.save("cropped_224.png")

        # Predict
        results = predict_image(self.model, "cropped_224.png", top_k=3)
        msg = "\n".join([f"{cls}: {prob:.2%}" for cls, prob in results])
        messagebox.showinfo("Prediction", msg)

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    model = load_model(MODEL_PATH)

    root = tk.Tk()
    app = CropperApp(root, model, "captured_image.jpg")  # Replace with your image or captured frame
    root.mainloop()

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import traceback
import os
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import time

# ---------------------------
# Config & Setup
# ---------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

MODEL_PATH = "models/efficientnet_parasite_final.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "ascaris_lumbricoides",
    "enterobius_vermicularis",
    "hookworm",
    "trichuris_trichiura",
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------
# Camera Utility
# ---------------------------
class CameraManager:
    """
    Manages camera access, trying multiple methods for Raspberry Pi 5 compatibility.
    """
    def __init__(self):
        self.cap = None

    def capture_frame(self):
        """
        Attempts to capture a single frame from the camera.
        Returns: (success, frame_bgr)
        """
        # Method 1: Try Standard OpenCV (V4L2)
        # On Pi 5 with libcamera-compat, this often works for index 0
        print("Set up camera...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Warmup
            print("Camera opened, warming up...")
            for _ in range(5):
                ret, frame = cap.read()
            
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                print("Capture successful via standard OpenCV.")
                return True, frame
            else:
                print("Standard OpenCV opened but returned empty frame.")

        # Method 2: GStreamer Pipeline (Optimized for Libcamera)
        # libcamerasrc -> appsink
        print("Trying GStreamer pipeline...")
        gst_pipe = (
            "libcamerasrc ! video/x-raw, width=1920, height=1080, framerate=30/1 ! "
            "videoconvert ! videoscale ! video/x-raw, format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                print("Capture successful via GStreamer.")
                return True, frame
        
        print("GStreamer failed.")
        return False, None

# ---------------------------
# Model Functions
# ---------------------------
def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    base = EfficientNetB0(include_top=False, weights=None, input_shape=input_shape, pooling="avg")
    outputs = layers.Dense(num_classes, activation="softmax")(base.output)
    model = models.Model(inputs=base.input, outputs=outputs)
    return model

def load_trained_model(model_path: str):
    if not os.path.exists(model_path):
        messagebox.showerror("Error", f"Model not found at {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"⚠️ Load failed, attempting to build and load weights: {e}")
        try:
            model = build_model()
            model.load_weights(model_path)
            return model
        except Exception as e2:
            messagebox.showerror("Error", f"Failed to load model: {e2}")
            return None

def predict_cv_image(model, cv_img: np.ndarray, top_k: int = 3):
    # Preprocess
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array) # EfficientNet preprocessing [0, 255] -> scaled
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    
    # Safely handle cases where strict top_k usage might exceed bounds if classes < k
    real_k = min(top_k, len(CLASS_NAMES))
    results = [(CLASS_NAMES[i], float(preds[i])) for i in top_indices[:real_k]]
    return results

# ---------------------------
# Main App Class
# ---------------------------
class ParasiteApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Parasite AI Classifier")
        self.geometry("1024x600") # Default for dev, will be fullscreen
        
        # FULLSCREEN MODE
        self.attributes("-fullscreen", True)
        
        # Key Bindings
        self.bind("<Escape>", lambda e: self.destroy()) # Exit on Esc

        # Load Model
        self.model = load_trained_model(MODEL_PATH)
        
        # Camera Manager
        self.cam_manager = CameraManager()

        # State
        self.current_image = None
        self.tk_img_ref = None
        self.rect_id = None
        self.start_x, self.start_y = None, None
        self.crop_box_size = 224
        self.cropped_image = None
        self.img_scale = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0

        # Layout Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_sidebar()
        self.setup_main_area()

    def setup_sidebar(self):
        # Wider sidebar for easier touch
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1) # Spacer

        # Fonts
        title_font = ctk.CTkFont(size=28, weight="bold")
        btn_font = ctk.CTkFont(size=20, weight="bold")
        lbl_font = ctk.CTkFont(size=16)

        # Title
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Parasite AI", font=title_font)
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # Buttons (Taller and with more padding)
        btn_height = 60
        
        self.btn_open = ctk.CTkButton(self.sidebar, text="📂 Open Image", command=self.open_image, font=btn_font, height=btn_height)
        self.btn_open.grid(row=1, column=0, padx=20, pady=15, sticky="ew")

        self.btn_capture = ctk.CTkButton(self.sidebar, text="📸 Capture", command=self.capture_image, font=btn_font, height=btn_height)
        self.btn_capture.grid(row=2, column=0, padx=20, pady=15, sticky="ew")

        self.btn_classify = ctk.CTkButton(self.sidebar, text="🤖 Classify", command=self.classify_image, 
                                          fg_color="green", hover_color="darkgreen", font=btn_font, height=btn_height)
        self.btn_classify.grid(row=3, column=0, padx=20, pady=15, sticky="ew")

        # Instructions
        self.lbl_instr = ctk.CTkLabel(self.sidebar, text="1. Select Image\n2. Drag Box\n3. Classify\n[ESC to Exit]", 
                                      text_color="gray", font=lbl_font)
        self.lbl_instr.grid(row=5, column=0, padx=20, pady=20, sticky="s")

        # Exit Button
        self.btn_exit = ctk.CTkButton(self.sidebar, text="❌ Exit", command=self.destroy, 
                                      fg_color="red", hover_color="darkred", font=btn_font, height=btn_height)
        self.btn_exit.grid(row=6, column=0, padx=20, pady=30, sticky="ew")

    def setup_main_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Canvas for Image
        self.canvas_frame = ctk.CTkFrame(self.main_frame)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        
        self.canvas.bind("<Button-1>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.update_crop)
        self.canvas.bind("<ButtonRelease-1>", self.finish_crop)

        # Results Area (Larger text)
        result_font = ctk.CTkFont(size=24, weight="bold")
        self.results_label = ctk.CTkLabel(self.main_frame, text="Load image & select area.", 
                                          font=result_font, wraplength=600)
        self.results_label.grid(row=1, column=0, sticky="ew", pady=10)

    # ---------------------------
    # Logic
    # ---------------------------
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.show_image(self.current_image)
                self.results_label.configure(text="Image loaded. Drag to crop.")
            else:
                messagebox.showerror("Error", "Failed to load image.")

    def capture_image(self):
        self.results_label.configure(text="Capturing... Please wait.")
        self.update_idletasks()
        
        # Run in thread to avoid freezing UI
        def _capture():
            success, frame = self.cam_manager.capture_frame()
            if success:
                self.after(0, lambda: self._on_capture_success(frame))
            else:
                self.after(0, self._on_capture_fail)
        
        threading.Thread(target=_capture, daemon=True).start()

    def _on_capture_success(self, frame):
        self.current_image = frame
        self.show_image(self.current_image)
        self.results_label.configure(text="Photo Captured. Drag to crop.")

    def _on_capture_fail(self):
        # Fallback to Mock
        self.current_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.current_image, "Camera Failed - Mock Mode", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.show_image(self.current_image)
        messagebox.showwarning("Camera Error", "Could not access camera (tried OpenCV & GStreamer).\nUsing Mock Image.")

    def show_image(self, cv_img):
        # Resize to fit canvas while maintaining aspect ratio
        h, w = cv_img.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10: # Fallback if canvas not rendered yet
            canvas_w, canvas_h = 600, 400

        scale = min(canvas_w/w, canvas_h/h, 1.0) # Don't scale up too much if small
        # Actually, let's allow scaling up to fill available space
        scale = min(canvas_w/w, canvas_h/h)
        
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(cv_img, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        self.tk_img_ref = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        self.canvas.delete("all")
        
        # Center image
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_img_ref)
        
        # Store scale/offsets for cropping mapping
        self.img_scale = scale
        self.img_offset_x = x_offset
        self.img_offset_y = y_offset

    def start_crop(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def update_crop(self, event):
        if self.start_x and self.start_y:
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            
            # Fixed size box logic:
            # We want a literal 224x224 patch from the SOURCE image.
            # Convert 224 source pixels -> canvas pixels
            box_display_size = int(self.crop_box_size * self.img_scale)
            
            x1, y1 = self.start_x, self.start_y
            x2, y2 = x1 + box_display_size, y1 + box_display_size
            
            self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="cyan", width=2)

    def finish_crop(self, event):
        if self.current_image is None or not hasattr(self, 'img_scale'):
            return
            
        x1_canvas = self.start_x
        y1_canvas = self.start_y
        
        # Map back to image coordinates
        # (canvas_x - offset) / scale = image_x
        ix1 = int((x1_canvas - self.img_offset_x) / self.img_scale)
        iy1 = int((y1_canvas - self.img_offset_y) / self.img_scale)
        
        ix2 = ix1 + self.crop_box_size
        iy2 = iy1 + self.crop_box_size
        
        # Clip to bounds
        h, w = self.current_image.shape[:2]
        
        # Ensure we don't crash if out of bounds, just clamp
        if ix1 < 0: ix1 = 0
        if iy1 < 0: iy1 = 0
        if ix2 > w: ix2 = w
        if iy2 > h: iy2 = h
        
        # Check valid size
        if ix2 > ix1 and iy2 > iy1:
            self.cropped_image = self.current_image[iy1:iy2, ix1:ix2]
            self.results_label.configure(text="Area selected. Ready to classify.")
        else:
             self.results_label.configure(text="Invalid crop area (outside image). Try again.")

    def classify_image(self):
        if self.cropped_image is None:
            messagebox.showwarning("Warning", "Please select an area (crop) first!")
            return
            
        if self.model is None:
             messagebox.showerror("Error", "Model not loaded!")
             return

        try:
            results = predict_cv_image(self.model, self.cropped_image)
            
            if not results:
                self.results_label.configure(text="No results.")
                return

            # Format results
            top_class, top_score = results[0]
            text = f"Result: {top_class}\nConfidence: {top_score:.2%}"
            
            # Color code based on confidence
            color = "green" if top_score > 0.8 else "orange" if top_score > 0.5 else "red"
            self.results_label.configure(text=text, text_color=color)
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Classification failed: {e}")

if __name__ == "__main__":
    app = ParasiteApp()
    app.mainloop()

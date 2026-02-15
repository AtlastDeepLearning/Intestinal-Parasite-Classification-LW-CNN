import tensorflow as tf
import os
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
import sys
import time

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "models/efficientnet_b0_parasite.h5" # Updated path
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
            print("Loading weights into new model structure due to shape mismatch...")
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
        self.current_image = None   # OpenCV image (BGR) - High Res
        self.display_image = None   # PIL Image - Scaled for display
        self.tk_img_ref = None
        self.rect_id = None
        self.start_x, self.start_y = None, None
        self.crop_box_size = 224    # fixed 224x224 crop box (model input size)
        
        self.camera = None
        self.is_camera_running = False
        self.camera_thread = None

        # Camera Configuration
        # We use a 640x480 preview which fits perfectly in our left pane
        self.preview_size = (640, 480) 
        self.capture_size = (1920, 1080) # Capture high res, but preview is smaller
        
        # UI Setup
        self.setup_ui()
        
        # Initialize Camera if available
        self.init_camera()
        
        # Auto-start camera
        if self.camera:
            self.toggle_camera()

    def setup_ui(self):
        # ---------------------------------------------------------
        # Main Window Config
        # ---------------------------------------------------------
        # Target Resolution: 800x480 (Official RPi 7" Touchscreen)
        # Layout: 
        #   Left:  640x480 Camera/Image Area
        #   Right: 160x480 Control Panel
        # ---------------------------------------------------------
        
        # AGGRESSIVE CENTERING STRATEGY
        # 1. Remove window decorations (title bar, borders) entirely
        self.root.overrideredirect(True) 
        
        # 2. Force exact geometry at 0,0
        self.root.geometry("800x480+0+0")
        
        # 3. Force window manager to update
        self.root.update_idletasks()
        
        self.root.configure(bg="#121212") # Dark background

        # Bind Escape key to exit (since we have no title bar)
        self.root.bind("<Escape>", lambda e: self.close_app())

        # Main Container (Horizontal Layout)
        main_frame = tk.Frame(self.root, bg="#121212")
        main_frame.pack(fill="both", expand=True)

        # ---------------------------
        # Left Panel (Camera/Canvas)
        # ---------------------------
        # Fixed width of 640 to match 4:3 aspect ratio and camera preview
        self.left_panel = tk.Frame(main_frame, bg="black", width=640, height=480)
        self.left_panel.pack(side="left", fill="y")
        self.left_panel.pack_propagate(False) # Enforce size

        self.canvas_width = 640
        self.canvas_height = 480
        
        self.canvas = tk.Canvas(
            self.left_panel, 
            bg="black", 
            width=self.canvas_width, 
            height=self.canvas_height,
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.update_crop)
        self.canvas.bind("<ButtonRelease-1>", self.finish_crop)

        # ---------------------------
        # Right Panel (Controls)
        # ---------------------------
        self.right_panel = tk.Frame(main_frame, bg="#1E1E1E", width=160)
        self.right_panel.pack(side="right", fill="both", expand=True)
        self.right_panel.pack_propagate(False) # Enforce size if possible

        # Padding for buttons
        PAD_X = 10
        PAD_Y = 10
        BTN_HEIGHT = 2
        
        # Style logic
        btn_style = {"font": ("Arial", 11, "bold"), "bd": 0, "relief": "flat"}

        # 1. Status / Info at the top
        self.results_var = tk.StringVar(value="Ready")
        self.lbl_status = tk.Label(
            self.right_panel, 
            textvariable=self.results_var, 
            font=("Arial", 10), 
            bg="#1E1E1E", 
            fg="#00E676", # Green text
            wraplength=140,
            justify="center"
        )
        self.lbl_status.pack(pady=(20, 10), padx=5, fill="x")

        # Separator
        tk.Frame(self.right_panel, bg="#333", height=2).pack(fill="x", padx=10, pady=5)

        # 2. Camera Button
        self.btn_camera = tk.Button(
            self.right_panel, 
            text="Start Camera", 
            command=self.toggle_camera, 
            bg="#2979FF", # Blue
            fg="white",
            activebackground="#1565C0",
            activeforeground="white",
            height=BTN_HEIGHT,
            **btn_style
        )
        self.btn_camera.pack(fill="x", padx=PAD_X, pady=PAD_Y)

        # 3. Snap Button
        self.btn_snap = tk.Button(
            self.right_panel, 
            text="Capture", 
            command=self.snap_image, 
            bg="#FFC107", # Amber
            fg="black",
            activebackground="#FFA000",
            activeforeground="black",
            height=BTN_HEIGHT,
            state="disabled",
            **btn_style
        )
        self.btn_snap.pack(fill="x", padx=PAD_X, pady=PAD_Y)

        # 4. Classify Button
        self.btn_classify = tk.Button(
            self.right_panel, 
            text="Analyze", 
            command=self.classify_image, 
            bg="#00E676", # Green
            fg="black",
            activebackground="#00C853",
            activeforeground="black",
            height=BTN_HEIGHT,
            **btn_style
        )
        self.btn_classify.pack(fill="x", padx=PAD_X, pady=PAD_Y)

        # Spacer to push Exit to bottom
        tk.Frame(self.right_panel, bg="#1E1E1E").pack(fill="both", expand=True)

        # 5. Exit Button
        self.btn_exit = tk.Button(
            self.right_panel, 
            text="Exit", 
            command=self.close_app, 
            bg="#D32F2F", # Red
            fg="white",
            activebackground="#C62828",
            activeforeground="white",
            height=BTN_HEIGHT,
            **btn_style
        )
        self.btn_exit.pack(fill="x", padx=PAD_X, pady=PAD_Y + 10)

    def close_app(self):
        self.stop_camera()
        self.root.quit()

    def init_camera(self):
        try:
            # Check for Picamera2 availability
            try:
                global Picamera2
                
                # 🔹 Add Picamera2 (with system path fallback for venv)
                try:
                    from picamera2 import Picamera2
                except ImportError:
                    # If in venv without system-site-packages, try adding system dist-packages manually
                    import sys
                    sys.path.append('/usr/lib/python3/dist-packages')
                    try:
                        from picamera2 import Picamera2
                    except ImportError:
                        print("Warning: Picamera2 library not found even in system paths.")
                        raise ImportError("Picamera2 not found")

                self.camera = Picamera2()
                # Configure for video/preview to minimize latency
                config = self.camera.create_video_configuration(main={"size": self.preview_size, "format": "RGB888"})
                self.camera.configure(config)
                self.camera.start() # Start hardware, but not capturing yet
                print("Camera initialized successfully.")
            except ImportError:
                print("Picamera2 not found. Camera features disabled.")
                self.camera = None
                self.btn_camera.config(state="disabled", text="No Camera", bg="#555555")
            except Exception as e:
                print(f"Camera init failed: {e}")
                self.camera = None
                self.btn_camera.config(state="disabled", bg="#555555")

        except Exception as e:
            print(f"Error initializing camera: {e}")

    def toggle_camera(self):
        if not self.is_camera_running:
            # START Camera
            self.is_camera_running = True
            # Update to Stop Styling
            self.btn_camera.config(text="Stop Camera", bg="#D32F2F", activebackground="#C62828") 
            self.btn_snap.config(state="normal", bg="#FFC107")
            self.btn_classify.config(state="disabled", bg="#555555")
            self.rect_id = None 
            
            # Start preview loop
            self.update_preview()
        else:
            # STOP Camera (Cancel)
            self.stop_camera()

    def stop_camera(self):
        self.is_camera_running = False
        # Update to Start Styling
        self.btn_camera.config(text="Start Camera", bg="#2979FF", activebackground="#1565C0")
        self.btn_snap.config(state="disabled", bg="#555555")
        self.btn_classify.config(state="normal", bg="#00E676")

    def update_preview(self):
        if self.is_camera_running and self.camera:
            try:
                # Capture recent frame
                # stream='main' ensures we get the resizing/format we configured
                frame = self.camera.capture_array('main') 
                
                if frame is None:
                     print("Frame is None!")
                     self.root.after(30, self.update_preview)
                     return

                # Debug: Check if frame is black
                mean_val = frame.mean()
                if mean_val < 1.0:
                    print(f"Warning: Frame is black! Shape: {frame.shape}, Mean: {mean_val:.2f}")
                
                # Convert to PIL for display
                img = Image.fromarray(frame) # RGB
                
                # Resize to fit canvas
                img_display = img.resize((self.canvas_width, self.canvas_height))
                self.tk_img_ref = ImageTk.PhotoImage(img_display)
                
                self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img_ref)
                
                # Keep raw frame as current_image (converted to BGR for OpenCV consistency)
                self.current_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            except Exception as e:
                print(f"Preview error: {e}")
                
            # Schedule next frame
            self.root.after(30, self.update_preview)

    def snap_image(self):
        """Freeze the current frame for processing."""
        if self.camera and self.is_camera_running:
            # Capture one final high-quality frame if possible, or just use the last preview frame
            self.stop_camera()
            self.results_var.set("Image Snapped. Draw a box to crop.")
            
            # Ensure we show the static image clearly
            if self.current_image is not None:
                self.show_image(self.current_image)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.show_image(self.current_image)
            self.results_var.set("Image loaded. Draw a box to crop.")

    def show_image(self, cv_img):
        """Display OpenCV image on Canvas, scaling to fit."""
        if cv_img is None: return

        # Convert BGR -> RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Scale to fit canvas while maintaining aspect ratio
        w, h = pil_img.size
        # Avoid division by zero
        if w == 0 or h == 0: return

        scale = min(self.canvas_width/w, self.canvas_height/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        self.display_image = pil_img.resize((new_w, new_h))
        self.tk_img_ref = ImageTk.PhotoImage(self.display_image)
        
        self.canvas.delete("all")
        # Center the image
        x_offset = (self.canvas_width - new_w) // 2
        y_offset = (self.canvas_height - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_img_ref)
        
        # Store scale/offsets for coordinate mapping
        self.img_scale = scale
        self.img_x_offset = x_offset
        self.img_y_offset = y_offset

    def start_crop(self, event):
        if self.current_image is None or self.is_camera_running: return
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id: self.canvas.delete(self.rect_id)
        
    def update_crop(self, event):
        if self.current_image is None or self.is_camera_running: return
        if not self.start_x: return 
        
        # Draw dynamic box (visual feedback only)
        # We enforce fixed size at the end, but let user drag to position
        if self.rect_id: self.canvas.delete(self.rect_id)
        
        # We need to calculate how big 224 pixels is on the SCALED display image
        if hasattr(self, 'img_scale'):
            display_box_size = int(self.crop_box_size * self.img_scale)
            
            x1 = self.start_x
            y1 = self.start_y
            
            # Center the box on the mouse? Or top-left?
            # User workflow: square off parasite.
            # Let's draw from top-left (click point)
            x2 = x1 + display_box_size
            y2 = y1 + display_box_size
            
            self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def finish_crop(self, event):
        if self.current_image is None or self.is_camera_running: return
        pass # The box stays visible

    def classify_image(self):
        if self.current_image is None:
            self.show_custom_popup("Warning", "No image to classify!", is_error=True)
            return
        
        if not self.rect_id:
            self.show_custom_popup("Warning", "Please draw a crop box first!", is_error=True)
            return

        try:
            # Get coordinates from canvas
            x1, y1, x2, y2 = self.canvas.coords(self.rect_id)
            
            # Map back to original image coordinates
            # Remove offsets
            img_x = x1 - self.img_x_offset
            img_y = y1 - self.img_y_offset
            
            # Unscale
            if hasattr(self, 'img_scale') and self.img_scale > 0:
                real_x = int(img_x / self.img_scale)
                real_y = int(img_y / self.img_scale)
                
                # Fixed 224x224 size
                real_w = self.crop_box_size
                real_h = self.crop_box_size
                
                # Crop from original high-res image
                roi = self.current_image[real_y:real_y+real_h, real_x:real_x+real_w]
                
                
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    self.show_custom_popup("Error", "Selected area is empty!", is_error=True)
                    return

                # If crop is smaller than 224 (edge of image), pad it
                if roi.shape[0] < self.crop_box_size or roi.shape[1] < self.crop_box_size:
                     full_roi = np.zeros((self.crop_box_size, self.crop_box_size, 3), dtype=np.uint8)
                     full_roi[:roi.shape[0], :roi.shape[1]] = roi
                     roi = full_roi
                
                # Predict
                results = predict_cv_image(self.model, roi, top_k=3)
                
                # Format Results for large popup
                top_result = results[0]
                res_text = f"{top_result[0]}\n{top_result[1]*100:.1f}% Confidence\n\n"
                
                # Add secondary results smaller
                if len(results) > 1:
                    res_text += "Alternatives:\n"
                    for i in range(1, len(results)):
                         res_text += f"{results[i][0]}: {results[i][1]*100:.1f}%\n"
                
                self.results_var.set(f"Top: {results[0][0]} ({results[0][1]*100:.1f}%)")
                
                # Show Custom Popup
                self.show_custom_popup("Analysis Complete", res_text)

        except Exception as e:
            traceback.print_exc()
            self.show_custom_popup("Error", f"Classification failed:\n{e}", is_error=True)

    def show_custom_popup(self, title, message, is_error=False):
        """
        Creates a large, touch-friendly overlay popup.
        """
        # Create overlay frame (covers everything)
        overlay = tk.Frame(self.root, bg="#000000")
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Container for content (centered)
        content_frame = tk.Frame(overlay, bg="#1E1E1E", padx=40, pady=40)
        content_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Title
        lbl_title = tk.Label(
            content_frame, 
            text=title, 
            font=("Arial", 24, "bold"), 
            fg="#FF5252" if is_error else "#00E676",
            bg="#1E1E1E"
        )
        lbl_title.pack(pady=(0, 20))
        
        # Message
        lbl_msg = tk.Label(
            content_frame, 
            text=message, 
            font=("Arial", 18), 
            fg="white",
            bg="#1E1E1E",
            wraplength=600,
            justify="center"
        )
        lbl_msg.pack(pady=(0, 30))
        
        # Close Button
        btn_close = tk.Button(
            content_frame,
            text="CLOSE",
            font=("Arial", 20, "bold"),
            bg="#2979FF",
            fg="white",
            activebackground="#1565C0",
            activeforeground="white",
            width=10,
            height=2,
            command=overlay.destroy 
        )
        btn_close.pack()


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    if "picamera2" in sys.modules:
        pass # Already imported
        
    print("Loading AI Model...")
    # Ensure model path is correct relative to execution dir
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model not found at {MODEL_PATH}")
    
    model = load_model(MODEL_PATH)
    print("Model Loaded.")

    root = tk.Tk()
    root.title("Parasite Classifier (Pi 5)")
    root.geometry("800x480+0+0") # Force top-left on init too

    app = CropperApp(root, model)

    root.mainloop()
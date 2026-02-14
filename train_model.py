import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
import numpy as np

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # You can adjust this
DATASET_DIR = "parasite_dataset"
MODEL_SAVE_PATH = "models/efficientnet_parasite_final.keras"
TFLITE_SAVE_PATH = "models/parasite_model.tflite"

def main():
    print(f"TensorFlow Version: {tf.__version__}")
    
    # 1. Load Data
    print("Loading dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    
    class_names = train_ds.class_names
    print(f"Detected Classes: {class_names}")
    
    # optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # 2. Build Model (EfficientNetB0)
    print("Building EfficientNetB0 model...")
    base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False # Freeze base model initially
    
    model = models.Sequential([
        # Explicit input layer to ensure shape is defined for TFLite
        layers.Input(shape=(224, 224, 3)),
        # Preprocessing (EfficientNet expects 0-255 inputs but handles scaling internally if using keras application? 
        # Actually EfficientNetB0 includes rescaling logic in the model itself usually, BUT 
        # tf.keras.applications.efficientnet.preprocess_input does scaling.
        # Let's add a Rescaling layer to be safe and self-contained for TFLite)
        layers.Rescaling(1./255), # Normalize to [0,1] if model expects it, mostly EfficientNet expects specific scaling.
        # Wait, EfficientNetB0 from tf.keras.applications expects [0, 255] inputs if include_preprocessing=True (default in newer TF).
        # In older TF/Keras, it might not.
        # Let's use the explicit preprocessing layer from keras if available or just raw.
        # Safest for TFLite: Include generic normalization.
        # Actually, let's look at the docs: "EfficientNet models expect their inputs to be float tensors of pixels with values in the [0, 255] range."
        # So we do NOT need Rescaling(1./255) for the base model itself if we use the default.
        # However, for TFLite stability, lets just stick to the specific preprocessing.
        
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # 3. Train
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
        ]
    )
    
    print("Training finished.")
    
    # 4. Convert to TFLite
    print(f"Converting to TFLite: {TFLITE_SAVE_PATH}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable optimizations (quantization) for Pi 5
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print("✅ TFLite conversion complete.")
    print(f"Model saved to {TFLITE_SAVE_PATH}")
    print(f"Class names: {class_names}")

if __name__ == "__main__":
    main()

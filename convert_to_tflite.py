import tensorflow as tf

MODEL_PATH = "models/efficientnet_parasite_final.keras"
TFLITE_PATH = "models/parasite_model.tflite"

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved to {TFLITE_PATH}")

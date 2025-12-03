import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Config
MODEL_PATH = "../models/efficientnet_parasite_final.keras"
DATA_DIR = "../parasite_dataset/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load Data
print("Loading validation data...")
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Important for confusion matrix
)

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# Predict
print("Generating predictions...")
y_pred_probs = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes
class_names = list(val_generator.class_indices.keys())

# Confusion Matrix
print("Computing confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Validation Set)')
plt.xticks(rotation=45)
plt.tight_layout()

# Save
save_path = "../plots/confusion_matrix.png"
plt.savefig(save_path)
print(f"Confusion matrix saved to {save_path}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

Intestinal-Parasite-Classification-LW-CNN

Image classification using a lightweight Convolutional Neural Network (EfficientNetB0 backbone) for intestinal parasite detection in microscopic images.

📌 Overview

This project implements a lightweight CNN based on EfficientNetB0 for classifying intestinal parasite eggs in microscopic stool images. The model is designed to be efficient, accurate, and suitable for deployment on limited-resource environments such as clinical labs or edge devices.

It supports 4 parasite categories:

Ascaris lumbricoides

Enterobius vermicularis

Hookworm eggs

Trichuris trichiura

📂 Dataset

Microscopic images of parasite eggs with varied resolutions (1920x1080, 1344x1080, etc.)

Images organized into train/validation/test directories:

dataset/
  train/
    ascaris_lumbricoides/
    enterobius_vermicularis/
    hookworms/
    trichuris_trichiura/
  val/
    ...
  test/
    ...

⚙️ Requirements
pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python pillow


(Optional for visualization/reporting)

pip install seaborn reportlab

🚀 Training

Run the training script or notebook:

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("efficientnet_parasite.h5", save_best_only=True, monitor="val_accuracy", mode="max")
earlystop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    train_gen_processed,
    validation_data=val_gen_processed,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop],
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE
)


The model can also be fine-tuned by unfreezing the last few layers of EfficientNet.

🧪 Evaluation

You can evaluate the model on a test set and generate classification reports, confusion matrices, and F1 scores:

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Metrics
print(classification_report(test_gen.classes, y_pred_classes, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(test_gen.classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.show()

📊 Results

Example small test set (40 images):

Accuracy: 75%
Macro Precision: 0.81
Macro Recall: 0.75
Macro F1: 0.72


(Update with full evaluation metrics once trained on the full dataset)

📁 Project Structure
Intestinal-Parasite-Classification-LW-CNN/
├── dataset/               # Train/Val/Test images
├── notebooks/             # Jupyter notebooks for training & evaluation
├── efficientnet_parasite.h5 # Saved best model
└── README.md

🔮 Future Work

Extend to object detection (YOLOv8 / EfficientDet) to handle images with multiple parasites.

Optimize for mobile/edge deployment (TFLite, ONNX).

Improve dataset balance and augmentation.

🤝 Contributing

Contributions are welcome! Open an issue or submit a PR.

📜 License

[Add your license here]

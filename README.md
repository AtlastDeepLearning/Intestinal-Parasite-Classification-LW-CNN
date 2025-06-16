# Intestinal-Parasite-Classification-LW-CNN
Image Classification using Lightweight Convolutional Neural Networks for Intestinal Parasite Detection

## Overview
This project implements a lightweight Convolutional Neural Network (CNN) for the classification of intestinal parasites in microscopic images. The model is designed to be efficient and accurate in identifying various types of parasites commonly found in stool samples.

## Dataset
The dataset contains microscopic images of various intestinal parasites with the following characteristics:
- 4 different parasite categories
- Images in various dimensions (primarily 1920x1080 and 1344x1080)
- Categories include:
  - Ascaris lumbricoides
  - Enterobius vermicularis
  - Hookworm egg
  - Trichuris trichiura

## Requirements
- Python 3.7+
- Linux OS (required for training)
- CUDA-capable GPU (recommended for training)
- Required Python packages:
  ```bash
  pip install torch torchvision
  pip install numpy pandas
  pip install pillow
  pip install reportlab  # for PDF report generation
  ```

## Project Structure
```
Intestinal-Parasite-Classification-LW-CNN/
├── data/
│   ├── images/          # Original images
│   └── annotations/     # JSON annotation files
├── src/
│   ├── models/         # CNN model definitions
│   ├── utils/          # Utility functions
│   └── training/       # Training scripts
├── notebooks/          # Jupyter notebooks
├── scripts/            # Helper scripts
│   ├── analyze_json.py # Dataset analysis tool
│   └── rename_images.py # Image renaming utility
└── README.md
```

## Usage

### Dataset Analysis
To analyze the dataset and generate a PDF report:
```bash
python scripts/analyze_json.py --pdf dataset_report.pdf
```

### Image Organization
To rename images based on their categories:
```bash
python scripts/rename_images.py test_labels_200.json path/to/images --output path/to/output
```

### Training
1. Ensure you're on a Linux system
2. Navigate to the project directory
3. Run the training notebook:
```bash
jupyter notebook notebooks/training.ipynb
```

## Model Architecture
The lightweight CNN architecture is designed to be:
- Efficient in terms of computational resources
- Suitable for deployment on edge devices
- Accurate in parasite classification
- Fast in inference time

## Results
[Add your model's performance metrics and results here]

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Add your license information here]

## Acknowledgments
- Dataset provided by [source]
- Based on research from [relevant papers/studies]

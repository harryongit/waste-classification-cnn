# Waste Segregation Using CNNs – Group Project Report

**Course:** ACPML – Waste Segregation Using CNNs [IITB]

## Group Members

- Harivdan Shrihari Narayanshastri
- Nagamani Yavagoni
- Suranjan Banerjee

---

## Table of Contents

1. [Objective](#1-objective)
2. [Dataset](#2-dataset)
3. [Data Pre-processing and Augmentation](#3-data-pre-processing-and-augmentation)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Conclusion](#6-conclusion)
7. [How to Run the Project](#7-how-to-run-the-project)

---

## 1. Objective

Design and implement an automated waste‑segregation system using Convolutional Neural Networks (CNNs) for multi‑class classification of common waste categories:

- Cardboard
- Glass
- Paper
- Plastic
- Food_Waste
- Metal
- Other

The goal is to improve sorting efficiency and support sustainable waste‑management by accurately classifying images into these categories.

## 2. Dataset

Local ZIP archive used directly in the notebook:

- Path: `\Dataset_Waste_Segregation.zip`
- Structure: one top‑level folder containing subfolders per class (as listed above)
- Splitting: performed programmatically with `validation_split=0.2`

No Kaggle download or API token is required.

## 3. Data Pre-processing and Augmentation

### Pre-processing

- Images resized to **224 × 224 × 3**
- Pixel values rescaled to **[0, 1]** using `layers.Rescaling(1/255)`
- Data pipelines created with `tf.keras.utils.image_dataset_from_directory`

### Data Augmentation (Training Only)

- Horizontal flip (`RandomFlip`)
- Small rotations (`RandomRotation(0.1)`)
- Zoom (`RandomZoom(0.1)`)

Validation data uses only resizing and rescaling.

## 4. Methodology

### Model

A compact CNN implemented in TensorFlow/Keras:

- Convolution → MaxPooling blocks: 16/32/64 filters, `padding="same"`, ReLU
- Flatten → Dropout (0.3) → Dense(128, ReLU)
- Final Dense with `num_classes` logits

### Training Configuration

- Loss: `SparseCategoricalCrossentropy(from_logits=True)`
- Optimizer: Adam
- Batch size: 32
- Epochs: 10
- Metrics: Accuracy
- Caching and prefetching enabled for efficient input pipelines

The model and pipeline follow the new notebook cells appended to `CNN_Assg_Waste_Segregation_Starter.ipynb`.

## 5. Results

Training prints epoch‑wise training/validation accuracy and loss. Final validation metrics are dataset‑dependent and are reported in the notebook output after `model.evaluate`.

The trained model is saved as:

- `waste_cnn.keras`

## 6. Conclusion

A simple, efficient CNN combined with lightweight augmentation achieves baseline performance for seven‑class waste classification using a local ZIP dataset and TensorFlow’s directory loaders. This setup provides a clean foundation for future improvements such as fine‑tuning deeper architectures, stronger regularization, or additional data.

## 7. How to Run the Project

1. Open the notebook: `CNN_Assg_Waste_Segregation_Starter.ipynb`.
2. Ensure the dataset ZIP exists at  
   `\Dataset_Waste_Segregation.zip`.
3. Run the cells sequentially:
   - Unzip and locate data directory
   - Build training/validation datasets
   - Define CNN and compile
   - Train and evaluate
   - Save the model (`waste_cnn.keras`)

All steps run locally without external downloads.

# Waste Segregation Using CNNs – Group Project Report

**Course:** ACPML – Waste Segregation Using CNNs

## Group Members

- Harivdan Shrihari Narayanshastri
- Nagamani Yavagoni
- Suranjan Banerjee

## GitHub Repository

https://github.com/harryongit/waste-classification-cnn

---

## Table of Contents

1. [Objective](#1-objective)
2. [Dataset](#2-dataset)
3. [Data Pre-processing and Augmentation](#3-data-pre-processing-and-augmentation)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Interpretation and Analysis](#6-interpretation-and-analysis)
7. [Conclusion](#7-conclusion)
8. [How to Run the Project](#8-how-to-run-the-project)

---

## 1. Objective

The objective of this group project is to design and implement an automated waste-segregation system using Convolutional Neural Networks (CNNs). The developed model classifies waste images into two primary categories:

- **O** – Organic / Biodegradable waste
- **R** – Recyclable / Non-organic waste

Automated segregation systems can significantly reduce manual sorting effort, improve recycling efficiency, and support sustainable waste-management practices. The proposed system demonstrates how deep learning can be applied in real-world environmental applications.

## 2. Dataset

**Source:** Kaggle – Waste Classification Data (techsash/waste-classification-data)  
The dataset was downloaded programmatically using the Kaggle API within Google Colab.

### Dataset Structure

- `/DATASET/TRAIN/O` and `/DATASET/TRAIN/R` – Training images
- `/DATASET/TEST/O` and `/DATASET/TEST/R` – Validation images

### Dataset Size

- **Training set:** 22,564 images (2 classes)
- **Validation set:** 2,513 images (2 classes)

The dataset consists of RGB photographs of real-world waste items captured under varying backgrounds, lighting conditions, and object orientations.

## 3. Data Pre-processing and Augmentation

### Pre-processing

- All images were resized to **224 × 224 × 3** to match the input requirements of the CNN backbone
- Pixel values were rescaled from [0, 255] to [0, 1] by dividing by 255

### Data Augmentation (Training Set Only)

To improve generalization and reduce overfitting, online augmentation was applied using ImageDataGenerator:

- Random rotation (±20°)
- Width/height shift (up to 20%)
- Shear and zoom (±20%)
- Horizontal flip

The validation set was not augmented; only rescaling was applied.

## 4. Methodology

### Model Selection – Transfer Learning with MobileNetV2

We employed **MobileNetV2**, pre-trained on ImageNet, as the feature-extraction backbone.

- Loaded with `include_top=False`
- All convolutional layers were frozen

This approach allows the model to leverage learned low-level and mid-level visual features while reducing training time and mitigating overfitting.

### Custom Classification Head

A lightweight classification head was added:

- **GlobalAveragePooling2D** – Converts feature maps into a 1280-dimensional vector
- **Dense (128, ReLU activation)** – Task-specific representation learning
- **Dropout (0.3)** – Regularization
- **Dense (2, Softmax activation)** – Output probabilities for classes O and R

### Training Configuration

- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (learning rate = 0.001)
- **Batch Size:** 32
- **Maximum Epochs:** 20

### Callbacks Used

- **EarlyStopping** (monitor = val_loss, patience = 5, restore_best_weights = True)
- **ReduceLROnPlateau** (monitor = val_loss, factor = 0.5, patience = 3, min_lr = 1e-7)

Training was conducted in Google Colab using a T4 GPU.

## 5. Results

### Training Summary (Early Stopping at Epoch 7)

- **Final Training Accuracy:** 95.6%
- **Final Validation Accuracy:** 81.5%
- **Training Loss:** 0.113
- **Validation Loss:** 0.414

The training curves indicate steady convergence. Validation loss increased slightly after several epochs, indicating mild overfitting.

### Confusion Matrix (Validation Set – 2,513 Images)

**Organic (O):**
- 1,319 / 1,401 correctly classified
- 94.15% class accuracy

**Recyclable (R):**
- 910 / 1,112 correctly classified
- 81.83% class accuracy

Most misclassifications involve recyclable items predicted as organic, likely due to visual similarities between certain packaging materials and organic waste.

### Classification Metrics

**Organic (O):**
- Precision: 0.867
- Recall: 0.941
- F1-score: 0.903

**Recyclable (R):**
- Precision: 0.917
- Recall: 0.818
- F1-score: 0.865

**Overall Performance:**
- Accuracy: 88.7%
- Macro F1-score: 0.884
- Weighted F1-score: 0.886

The model demonstrates high recall for organic waste and strong precision for recyclable waste, though recyclable recall can be further improved.

## 6. Interpretation and Analysis

The MobileNetV2-based transfer learning model performs effectively for binary waste classification. Organic waste is classified with higher recall, while recyclable waste shows slightly lower recall due to:

- Greater intra-class variability
- Visual overlap between recyclable and organic items
- Presence of soiled recyclable materials

The performance gap between training and validation accuracy suggests opportunities for improvement through:

- Fine-tuning upper MobileNetV2 layers with a lower learning rate
- Applying stronger regularization
- Increasing the diversity of recyclable training samples

Despite these limitations, an overall validation accuracy of approximately 89% demonstrates that lightweight CNN models are viable for practical, near-real-time waste segregation systems.

## 7. Conclusion

This group project demonstrates the practical application of deep learning and transfer learning techniques for automated waste segregation. Using MobileNetV2 as a feature extractor enabled efficient training and strong performance on a real-world dataset.

The developed model can serve as a foundational component in smart waste-management systems and can be further enhanced through fine-tuning and dataset expansion.

## 8. How to Run the Project

1. Open the Colab notebook from the GitHub repository:
   - `wastesegregation_iiitbproject.ipynb`

2. Enable GPU runtime in Google Colab

3. Upload your `kaggle.json` API token and execute the dataset download cell

4. Run all notebook cells sequentially:
   - Data loading and augmentation
   - Model construction and training
   - Performance visualization
   - Model saving

The notebook is fully reproducible and generates the results reported above when executed with the same dataset and settings.
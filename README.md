# Waste Segregation Using CNNs – Group Project Report

**Course:** ACPML – Waste Segregation Using CNNs [IITB]

## Group Members

- Harivdan Shrihari Narayanshastri
- Nagamani Yavagoni
- Suranjan Banerjee

A lightweight CNN baseline for multi‑class waste classification using a local ZIP dataset and TensorFlow/Keras.

## Overview

The notebook builds a simple CNN to classify images into seven classes:
Cardboard, Glass, Paper, Plastic, Food_Waste, Metal, Other.

## Dataset

- Local ZIP archive:  
  `\Dataset_Waste_Segregation.zip`
- Contains a single top‑level directory with subfolders per class.
- The notebook splits training/validation with `validation_split=0.2`.

No Kaggle download is required.

## Requirements

- Python 3.x
- TensorFlow (with Keras)
- NumPy

Install minimal dependencies:

```bash
pip install tensorflow numpy
```

## Run the Notebook

1. Open `CNN_Assg_Waste_Segregation_Starter.ipynb` in Jupyter or VS Code.
2. Ensure the ZIP path is set to:
   `\Dataset_Waste_Segregation.zip`
3. Run all cells sequentially:
   - Unzip and locate dataset
   - Create training/validation datasets
   - Build, train, and evaluate the CNN
   - Save the model

## Output

- Trained model file: `waste_cnn.keras`
- Console logs show training/validation accuracy and loss.

## Notes

- Augmentation uses random flip/rotation/zoom for better generalization.
- The pipeline relies on `image_dataset_from_directory` and `tf.data` prefetching for efficiency.

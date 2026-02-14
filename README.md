# Waste Segregation Using CNNs – Group Project Report

**Course:** ACPML – Waste Segregation Using CNNs

## Group Members

- Harivdan Shrihari Narayanshastri
- Nagamani Yavagoni
- Suranjan Banerjee


A deep learning project for waste classification using Convolutional Neural Networks (CNN).

## Project Overview

This project implements a CNN-based waste classifier that can automatically categorize waste items into different types. The model is trained to recognize and classify various waste materials to support waste segregation and recycling efforts.

## Project Structure

```
├── images/                          # Image data for training/testing
├── model/
│   └── waste_classifier_model.keras # Trained model file
├── notebook/
│   └── wastesegregation_iiitbproject.ipynb  # Jupyter notebook with analysis
├── output/
│   ├── classification_report.txt    # Model performance metrics
│   ├── waste_classifier_architecture.json   # Model architecture details
│   └── waste_classifier.weights.h5  # Model weights
├── python/
│   └── wastesegregation_iiitbproject.py    # Python implementation
└── README.md                        # This file
```

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- OpenCV (for image processing)
- Matplotlib (for visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/harryongit/waste-classification-cnn.git
cd waste-classification-cnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using the Jupyter Notebook
```bash
jupyter notebook notebook/wastesegregation_iiitbproject.ipynb
```

### Using the Python Script
```bash
python python/wastesegregation_iiitbproject.py
```

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input**: Waste item images
- **Output**: Classification of waste type

## Results

The model's performance metrics and classification report are available in the `output/` directory:
- Classification report: `output/classification_report.txt`
- Model architecture: `output/waste_classifier_architecture.json`

## Author

Created as part of the IIIT Bangalore project.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For more information, visit: https://github.com/harryongit/waste-classification-cnn

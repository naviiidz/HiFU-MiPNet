# HiFU-MiPNet: High-Frequency Ultrasound Material Identification and Size Estimation Pipeline

A deep learning-based system for material identification of microspheres using high-frequency ultrasound signals. This project implements both traditional machine learning and deep learning approaches for material classification.

## Project Structure

```
HiFU-MiPNet/
├── data/                          # Dataset directory
├── material_classification_ml/    # Traditional ML implementation
│   ├── config.yaml               # Configuration file
│   ├── train_ml.py              # Training script
│   ├── feature_extraction.py    # Feature extraction utilities
│   └── output/                  # Training outputs and results
├── material_classification_cnn/  # CNN-based implementation
│   ├── config.py                # Configuration settings
│   ├── model.py                 # CNN model architecture
│   ├── train.py                 # Training script
│   ├── predict.py               # Prediction script
│   ├── preprocessing.py         # Data preprocessing utilities
│   ├── peak_extract.py         # Peak extraction utilities
│   ├── visualize.py            # Visualization tools
│   └── final_model.h5          # Trained model weights
├── size_estimation_mlp/         # Size estimation implementation
├── manual_labeling/             # Manual labeling tools
├── LICENSE                      # MIT License
└── requirements.txt             # Project dependencies
```

## Data

The dataset can be downloaded from Google Drive:
[Download Dataset](https://drive.google.com/drive/folders/155tc1UMrrz98qi67ZMjPhE6f4ZXz_SgN?usp=sharing)

After downloading, place the data files in the `data/` directory.

## Features

- Multiple classification approaches:
  - Traditional Machine Learning (ML) with feature extraction
  - Deep Learning (CNN) with raw signal processing
- Comprehensive signal processing pipeline
- Peak extraction and analysis
- Material classification
- Size estimation capabilities
- Visualization tools for analysis
- Model evaluation metrics

## Requirements

### For ML Implementation
- Python 3.8+
- scikit-learn
- NumPy
- Pandas
- SciPy
- Matplotlib
- PyYAML

### For CNN Implementation
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- SciPy
- Matplotlib
- scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/naviiidz/HiFU-MiPNet.git
   cd HiFU-MiPNet
   ```

2. Install dependencies:
   ```bash
   # For ML implementation
   pip install -r material_classification_ml/requirements.txt
   
   # For CNN implementation and size estimation
   pip install -r material_classification_cnn/requirements.txt
   ```

## Usage

### Traditional ML Approach

```bash
cd material_classification_ml
python train_ml.py
```

### CNN-based Approach

```bash
cd material_classification_cnn
# Training
python train.py

# Prediction
python predict.py
```
### MLP size estimation

```
cd size_estimation_mlp
# Training
python train_size_model.py

# Prediction: Material + Size using bbox
python predict_mat_size.py
```

## Data Processing

The system processes ultrasound signals through the following steps:
1. Signal preprocessing and normalization
2. Peak extraction and analysis
3. Feature extraction (for ML approach)
4. Balancing
5. Train/test splitting

## Model Architectures

### Traditional ML
- Feature extraction from ultrasound signals
- Various ML classifiers (configurable)
- Cross-validation and hyperparameter tuning

### CNN Model
- 1D convolutional layers
- Batch normalization
- Max pooling
- Dropout for regularization
- Dense layers for classification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Navid Zarrabi (Navid.Zarrabi@torontomu.ca)

# Natural Disaster Hotspot Detection System 🌍

A Image Processing system for detecting natural disaster hotspots from satellite imagery using Convolutional Neural Networks (CNN) using Deep Learning.

## Overview

This project implements a machine learning solution for identifying natural disaster hotspots from satellite images. The system uses a CNN model trained on a dataset of hotspot and non-hotspot images to classify new images with high accuracy.

## Features

- 🔍 Deep learning-based image classification
- 📊 Real-time prediction with confidence scores
- 🎯 Adjustable decision threshold for fine-tuning predictions
- 📈 Model training with data augmentation
- 🌐 User-friendly web interface using Streamlit
- 📱 Support for multiple image formats (JPG, PNG, JPEG)

## Technical Details

### Model Architecture
- Convolutional Neural Network (CNN) with multiple layers
- Batch Normalization for improved training stability
- Dropout layers to prevent overfitting
- L2 regularization for better generalization
- Binary classification output (Hotspot/Non-Hotspot)

### Training Features
- Data augmentation for improved model robustness
- Class weight balancing for handling imbalanced datasets
- Early stopping to prevent overfitting
- Validation split for model evaluation
- Training metrics visualization

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install tensorflow opencv-python numpy matplotlib streamlit scikit-learn
```

## Usage

### Training the Model
Run the main script to train the model:
```bash
python main.py
```

### Web Interface
Launch the Streamlit web interface:
```bash
streamlit run frontend.py
```

## Project Structure

- `main.py` - Core model training and prediction logic
- `frontend.py` - Streamlit web interface
- `hotspot_classifier.h5` - Trained model file
- `dataset/` - Training and validation data
- `mean.py` - Additional utility functions

## Model Performance

The model includes several features to ensure optimal performance:
- Data augmentation to increase training data variety
- Class weight balancing to handle imbalanced datasets
- Early stopping to prevent overfitting
- Validation metrics tracking
- Adjustable prediction threshold

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

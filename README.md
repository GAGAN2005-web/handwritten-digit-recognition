# Handwritten Digit Recognition using Deep Neural Network

A Deep Learning project for recognizing handwritten digits (0–9) using the MNIST dataset and a Multi-Layer Perceptron (MLP). The project also compares different optimizers and analyzes the effect of Batch Normalization.

## 🚀 Features

- Handwritten digit classification using MNIST
- Multi-Layer Perceptron (MLP) neural network
- Comparison of SGD, RMSProp and Adam optimizers
- Comparison of models with and without Batch Normalization
- Interactive Streamlit web application
- Upload an image and predict the handwritten digit
- Display prediction confidence
- Test the model using MNIST samples

## 🧠 Model Architecture

The main MLP consists of:

- Input Layer: 784 neurons
- Dense Layer: 128 neurons with ReLU
- Dropout: 20%
- Dense Layer: 64 neurons with ReLU
- Output Layer: 10 neurons with Softmax

Total trainable parameters: **109,386**

## 📊 Dataset

The project uses the **MNIST handwritten digit dataset**.

- Training images: 60,000
- Testing images: 10,000
- Image size: 28 × 28 pixels
- Number of classes: 10 (digits 0–9)

### Preprocessing

1. Pixel values are normalized from 0–255 to 0–1.
2. Each 28 × 28 image is flattened into a 784-dimensional vector.
3. The processed data is provided to the MLP model.

## 📈 Results

### Main Model

| Metric | Result |
|---|---:|
| Test Accuracy | **97.71%** |
| Test Loss | **0.0745** |

### Optimizer Comparison

| Optimizer | Test Accuracy | Test Loss |
|---|---:|---:|
| SGD | 92.12% | 0.281438 |
| RMSProp | **97.29%** | 0.086071 |
| Adam | 97.07% | 0.092595 |

RMSProp achieved the highest accuracy among the three optimizers in this experiment.

### Batch Normalization Comparison

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| Without Batch Normalization | 97.36% | 0.084840 |
| With Batch Normalization | **97.53%** | **0.079312** |

Batch Normalization provided a small improvement in model performance.

## 🖥️ Streamlit Application

The project includes an interactive web application where users can:

- Upload a handwritten digit image
- Get the predicted digit
- View prediction confidence
- Test the model using MNIST images

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Pillow
- Streamlit

## 📂 Project Structure

```text
Handwritten-Digit-Recognition/
│
├── models/
│   └── mnist_mlp.keras
│
├── app.py
├── train.py
├── compare_optimizers.py
├── compare_regularizations.py
├── requirements.txt
└── README.md
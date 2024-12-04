# SVM-From-Scratch

A Python implementation of Support Vector Machines (SVM) built from the ground up, without relying on pre-built machine learning libraries. This project serves as both a learning tool and a demonstration of the fundamental concepts behind SVMs, including hard-margin and soft-margin approaches.

## Features

- **Hard-Margin SVM**: Implements a classifier for linearly separable data.
- **Soft-Margin SVM**: Introduces hinge loss for non-linearly separable data, balancing misclassification and margin maximization.
- **Gradient Descent Optimization**: Uses subgradient descent to minimize the SVM loss function.
- **In-Sample and Out-of-Sample Accuracies**: Includes training-testing data splits for evaluating model performance.

## Project Structure

- `main.py`: All functions for the SVM implementation, including:
  - Loss function calculation
  - Subgradient computation
  - Training algorithm
  - Accuracy evaluation
  - Data splitting for training and testing
- **Example Dataset**: Handwritten digit classification using a subset of the MNIST dataset to distinguish even and odd digits.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Libraries: `numpy`, `pandas`, `matplotlib`

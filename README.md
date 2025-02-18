# NN_assignment1_q3
**MNIST Classification with Adam & SGD Optimizers**

Overview

This project compares the performance of two optimizers, Adam and SGD, on the MNIST handwritten digit classification task. The accuracy trends of both optimizers are analyzed to understand their effectiveness.

Dataset

We use the MNIST dataset, which consists of 70,000 grayscale images (28x28 pixels) of handwritten digits (0-9). The dataset is split into:

Training Set: 60,000 images

Test Set: 10,000 images

Model Architecture

A simple neural network with:

Flatten layer (to convert 2D images into 1D vectors),
Dense layer with 128 neurons and ReLU activation,
Output layer with 10 neurons (softmax activation)

Training Details

Optimizer 1: Adam,
Optimizer 2: SGD,
Loss Function: Sparse Categorical Crossentropy,
Metrics: Accuracy,
Epochs: 10

Grad-CAM Reproducibility Project: CIFAR-10 & Synthetic Bias

This repository contains the implementation for Milestone 2 (MS2) of the Grad-CAM reproducibility project. It demonstrates the use of Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the decision-making process of a Convolutional Neural Network (CNN).

The project includes a specific experiment to detect "shortcut learning" (bias) by injecting synthetic artifacts into the training data.

Files

grad_cam_experiment.py: The main Python script containing the dataset generation, ResNet-18 model definition, Grad-CAM logic, and the training loop.

Milestone2_Report.md: The technical report detailing the methodology and preliminary results.

Requirements

To run this code, you need a Python environment with the following libraries installed:

PyTorch (torch, torchvision)

NumPy (numpy)

OpenCV (opencv-python)

Matplotlib (matplotlib)

You can install the dependencies via pip:

pip install torch torchvision numpy opencv-python matplotlib


Usage

You can run the entire experiment pipeline with a single command:

python grad_cam_experiment.py


Running on Google Colab

If you are using Google Colab, simply upload grad_cam_experiment.py to your session or copy the code into a cell. Ensure you enable the GPU runtime (Runtime > Change runtime type > T4 GPU) for faster training.

Experiments Overview

The script performs two sequential experiments:

1. Standard Experiment (Baseline)

Dataset: Standard CIFAR-10 ($32 \times 32$ images, 10 classes).

Model: ResNet-18 (Modified for small image size: 3x3 initial conv, no maxpool).

Goal: To verify that Grad-CAM correctly localizes semantic objects (e.g., the body of a Cat or the hull of a Ship) in a neutral setting.

Output: Visualizations of correctly classified images overlaid with attention heatmaps.

2. Biased Experiment (Bias Detection)

Dataset: Modified CIFAR-10.

Bias Injection: A bright yellow square ($6 \times 6$ pixels) is added to the top-left corner of every image in the 'Frog' class.

Goal: To test if the model learns to rely on the "yellow square" shortcut instead of the frog's biological features.

Output: Visualizations of 'Frog' predictions.

If the model is biased: The heatmap will highlight the yellow square in the corner.

If the model is robust: The heatmap will still focus on the frog's body.

Expected Output

During execution, you will see training logs for 5 epochs for each experiment. Upon completion, the script will generate and display (or save, depending on your environment) images showing:

The Original Image.

The Grad-CAM Heatmap overlaid on the image.

The True Class vs. Predicted Class.

Technical Implementation Details

Architecture: ResNet-18 modified for CIFAR-10 (stride 1 in first layer).

Target Layer: layer4[-1].conv2 (The last convolutional layer of the final residual block).

Training: 5 Epochs, SGD Optimizer (LR=0.01, Momentum=0.9).

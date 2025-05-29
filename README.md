# Segmentation of Pancreatic Ductal Adenocarcinoma using nnU-Net ResEnc with Tversky Loss

## Table of Contents
1.  [Overview](#overview)
2.  [Project Aim](#project-aim)
3.  [Prerequisites](#prerequisites)
4.  [Installation](#installation)
    * [Virtual Environment Setup](#virtual-environment-setup)
    * [Cloning the Repository](#cloning-the-repository)
    * [Installing Dependencies](#installing-dependencies)
5.  [Dataset Preparation](#dataset-preparation)
6.  [Preprocessing](#preprocessing)
7.  [Training](#training)
8.  [Testing](#testing)
9.  [References](#references)

## Overview
This repository provides the codebase and associated resources for the segmentation of Pancreatic Ductal Adenocarcinoma (PDAC). The methodology employs a modified nnU-Net architecture, incorporating Residual Encoder (ResEnc) blocks and utilizing a Tversky loss function.

## Project Aim
The primary objective of this model is to advance the accuracy of PDAC segmentation in medical imaging. A specific focus is placed on reducing the False Positive rates observed in the segmentation results reported by Liu et al. (2025) [1].

## Prerequisites
Ensure your system meets the following requirements before proceeding with the installation:
* **CUDA:** Version 11.1
* **cuDNN:** Version 9.0.0 (compatible with CUDA 12, though CUDA 11.1 is specified as the primary CUDA version)
    * *Note: Please verify cuDNN compatibility with CUDA 11.1 if CUDA 12.x is not the intended environment.*
* **Python:** Version 3.12

## Installation

### Virtual Environment Setup
It is highly recommended to create a dedicated virtual environment to manage project dependencies.

1.  **Create a Conda virtual environment:**
    ```bash
    conda create --name pdac_tversky python=3.12 -y
    ```
2.  **Activate the virtual environment:**
    ```bash
    conda activate pdac_tversky
    ```

### Cloning the Repository
1.  **Clone the project repository:**
    ```bash
    git clone https://github.com/mirklys/PDAC_Detection_forked.git
    ```
2.  **Navigate to the cloned directory:**
    ```bash
    cd PDAC_Detection_forked
    ```

### Installing Dependencies
1.  **Install primary requirements:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Install nnU-Net V2 (editable mode):**
    ```bash
    cd packages/nnunetv2
    pip install -e .
    ```
3.  **Install report-guided-annotation (editable mode):**
    ```bash
    cd ../report-guided-annotation
    pip install -e .
    ```
    *After this step, you might want to navigate back to the project's root directory, e.g., `cd ../..` if subsequent commands are expected to be run from there.*

## Dataset Preparation
Proper dataset organization is crucial for the nnU-Net framework.

1.  **Download the Dataset:**
    Acquire the dataset by following the instructions provided at the PANORAMA Grand Challenge [dataset link](https://panorama.grand-challenge.org/datasets-imaging-labels/).

2.  **Organize Dataset Files:**
    Place the downloaded dataset into the following directory structure within your `PDAC_Detection` project folder (you may need to create these directories if they don't exist):
    * **Training Images:**
        `workspace/nnUNet_raw/Dataset107_PDAC_detection/imagesTr/`
    * **Training Labels:**
        `workspace/nnUNet_raw/Dataset107_PDAC_detection/labelsTr/`

3.  **Create Additional Required Directories:**
    Ensure the following directories are also present for nnU-Net's operation:
    * `workspace/nnUNet_preprocessed/`
    * `workspace/nnUNet_results/`

## Preprocessing
The nnU-Net framework typically involves a preprocessing step to prepare the raw data.

1.  **Run the preprocessing script:**
   ```bash
   nnUNetv2_preprocess -d 107 -c 3d_fullres -np 1
   ```

## Training
Once the dataset is prepared and preprocessed, you can proceed with model training.

1.  **Run the training script:**
    This command initiates training using the 3D full-resolution configuration with the custom `nnUNetTrainerV2_ResEnc_TverskyLoss` for dataset 107. The `--npz` flag indicates that the preprocessed data is saved in `.npz` format.
    ```bash
    nnUNetv2_train 107 3d_fullres 0 -tr nnUNetTrainerTverskyLoss --c --npz
    ```

## Testing
To evaluate the trained model, use the provided testing script.

1.  **Run the testing script:**
    ```bash
    ./tversky_test.sh
    ```

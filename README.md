# CNN Geometric Shape Classifier

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

A robust Convolutional Neural Network (CNN) implemented in PyTorch, designed from scratch to classify geometric shapes. This system accurately distinguishes between empty segments, circles, and crosses—the foundational visual elements of games such as Tic-Tac-Toe.

## 🎯 Project Overview

This repository contains the architecture and training pipeline for a highly efficient custom CNN model. The model classifies images into three distinct categories:
- `blank`: Empty canvas space.
- `circle`: Contains a hand-drawn or digital circle (⭕).
- `nought`: Contains a hand-drawn or digital cross (❌).

Through extensive training, the model autonomously extracts deep visual features, achieving exceptional accuracy across test subsets.

---

## 🚀 Installation & Setup

### 1. Prerequisites

Ensure you have a Python environment set up. Install the required deep learning dependencies via `pip`:

```bash
pip install -r requirements.txt
```

### 2. Dataset Configuration

The data is separated into training and testing ZIP archives. To prepare your pipeline:
1. Extract `Blanck-Circle-Nought-dataset-trian.zip` and place its contents into `data/train/`.
2. Extract `Blanck-Circle-Nought-datase-test.zip` and place its contents into `data/test/`.

Your working directory must reflect the following structure:
```text
data/
├── train/
│   ├── blank/
│   ├── circle/
│   └── nought/
└── test/
    ├── blank/
    ├── circle/
    └── nought/
```

### 3. Execution

Launch the core Jupyter Notebook to begin training:
```bash
jupyter notebook CNN_classification_blanck_circle_nought.ipynb
```
> **Note:** If you are executing the pipeline locally (outside of Google Colab), update the dataset root directory path within the notebook to point to your local `./data/` directory.

---

## 🧠 Model Architecture

The custom CNN architecture (`CustomCNN`) is structured for maximal parameter efficiency while preventing overfitting via spatial dropouts:

| Layer Type | Configuration | Output Dimensions |
|------------|---------------|-------------------|
| **Input** | `3 channels (RGB)` | `3 x 28 x 28` |
| **Conv2D + MaxPool** | `in: 3, out: 16, kernel: 3x3` + `ReLU` | `16 x 14 x 14` |
| **Conv2D + MaxPool** | `in: 16, out: 32, kernel: 3x3` + `ReLU` | `32 x 7 x 7` |
| **Flatten** | `Flatten(start_dim=1)` | `1568` |
| **Linear (fc1)** | `in: 1568, out: 128` + `ReLU` + `Dropout(0.6)` | `128` |
| **Linear (fc2)** | `in: 128, out: 3` (Output) | `3` (Logits) |

---

## 📈 Training and Performance

The model was optimized using **Adam** (`lr=0.001`, `weight_decay=0.0005`) against a Multi-class **Cross-Entropy Loss** function.

### Metrics Summary

Through precisely 8 epochs of iterative training, the model achieves near-perfect classification generalization on unseen data.

| Dataset Phase | Metric | Final Result |
|---------------|--------|--------------|
| **Training** | Accuracy | `99.17%` |
| **Training** | Loss | `0.0289` |
| **Testing** | Accuracy | **`100.0%`** |
| **Testing** | Loss | **`0.0049`** |

---

**Author:** [Tareq Dorgam](https://github.com/tarekdorgam127-gif)  
*Built with passion and PyTorch.*

# 🧠 Brain Tumor MRI Classification

A deep learning project for **multi-class brain tumor classification from MRI images**, comparing a custom CNN with two transfer-learning models, **ResNet50** and **EfficientNetB0**.

The project also includes model evaluation, ensemble experiments, learning-rate experiments, and **Grad-CAM-based visual explanations** of model predictions.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes. It is not a medical diagnostic system and should not be used for clinical decision-making.

---

## 📌 Overview

Brain tumor classification from MRI images is a challenging computer vision problem due to variations in tumor appearance, size, location, and imaging characteristics.

This project investigates the ability of different deep learning architectures to classify brain MRI images into four categories:

* **Glioma**
* **Meningioma**
* **Pituitary tumor**
* **No tumor**

The project compares a custom CNN trained from scratch against pretrained architectures and examines their performance using several evaluation methods.

---

## 🎯 Objectives

The main goals of this project are to:

* Build a baseline CNN for MRI classification.
* Investigate transfer learning using **ResNet50** and **EfficientNetB0**.
* Compare the performance of different architectures.
* Analyze performance at both overall and class-specific levels.
* Experiment with model ensembles.
* Investigate learning-rate effects on model performance.
* Visualize model decisions using **Grad-CAM**.
* Explore the generalization and interpretability of deep learning models for MRI classification.

---

## 🗂️ Dataset

The project uses the **Brain Tumor MRI Dataset** available on Kaggle.

The dataset contains four classes:

| Class      | Description                             |
| ---------- | --------------------------------------- |
| Glioma     | MRI images containing glioma tumors     |
| Meningioma | MRI images containing meningioma tumors |
| Pituitary  | MRI images containing pituitary tumors  |
| No Tumor   | MRI images without a tumor              |

### Dataset

[Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Images are resized to **224 × 224 pixels** for model input.

---

## 🏗️ Models

Three primary architectures are investigated.

### 1. Custom CNN

A convolutional neural network designed specifically for this project.

The baseline CNN consists of multiple convolutional and pooling layers followed by fully connected layers for four-class classification.

The baseline model uses **grayscale MRI images** with pixel-value normalization.

### 2. ResNet50

A pretrained **ResNet50** architecture is used through transfer learning.

The original ImageNet classification head is replaced with a custom classification head for the four MRI classes.

Input images are processed as RGB images using the ResNet50 preprocessing pipeline.

### 3. EfficientNetB0

A pretrained **EfficientNetB0** model is evaluated as another transfer-learning approach.

The model uses:

* ImageNet pretrained weights
* `224 × 224 × 3` input
* Global Average Pooling
* Fully connected classification layers
* Dropout regularization
* Four-class softmax output

Fine-tuning is used to adapt the pretrained network to the MRI classification task.

---

## 🔬 Experimental Workflow

```text
                    Brain MRI Dataset
                           │
                           ▼
                 Image Preprocessing
                           │
                           ▼
                  Train / Validation
                       /        \
                      /          \
                     ▼            ▼
               Custom CNN    Transfer Learning
                              /           \
                             ▼             ▼
                         ResNet50    EfficientNetB0
                             \             /
                              \           /
                               ▼         ▼
                              Evaluation
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
            Metrics          Confusion          Grad-CAM
                             Matrix
                │
                ▼
          Model Comparison
                │
                ▼
       Ensemble Experiments
```

---

## 📊 Model Evaluation

Models are evaluated using multiple metrics rather than relying solely on accuracy.

The evaluation pipeline includes:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrices
* Classification reports
* ROC-AUC analysis where applicable
* Class-specific performance
* Training and validation loss
* Training and validation accuracy

This is important because overall accuracy can hide poor performance on individual tumor classes.

---

## 🔍 Grad-CAM Explainability

The project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to investigate which regions of an MRI image contribute to a model's prediction.

The generated heatmaps provide a visual representation of the areas receiving the strongest activation for a particular prediction.

This is used to investigate whether models are focusing on potentially meaningful regions of the MRI rather than irrelevant image features.

The current Grad-CAM implementation targets layers appropriate for each architecture, including:

```text
Custom CNN       → final convolutional layer
ResNet50         → conv5_block3_out
EfficientNetB0   → top_activation
```

Example:

```text
MRI Image
    │
    ▼
Trained Model
    │
    ▼
Predicted Class
    │
    ▼
Gradient Computation
    │
    ▼
Activation Weights
    │
    ▼
Grad-CAM Heatmap
    │
    ▼
Overlay on MRI
```

---

## 🤝 Ensemble Experiments

In addition to evaluating individual models, the project investigates whether combining model predictions can improve classification performance.

The repository contains ensemble-related experiments and learning-rate ensemble experiments.

The purpose is to determine whether different models make complementary predictions and whether combining them provides an advantage over individual architectures.

---

## 📈 Results

The current experiments demonstrate strong classification performance on the selected dataset, with test performance reaching approximately **92% accuracy** in the evaluated setup.

However, performance differs between classes and architectures.

In particular, **glioma classification presents greater difficulty** than some of the other classes, making class-specific evaluation important.

The repository includes additional evaluation outputs and experiments for examining these differences.

> **Important:** Performance on this dataset should not be interpreted as evidence of clinical reliability. A model can perform well on a benchmark dataset while still failing to generalize to MRI scans acquired from different institutions, scanners, populations, or imaging protocols.

---

## 🧪 Technologies

### Programming

* Python

### Deep Learning

* TensorFlow
* Keras
* CNN
* ResNet50
* EfficientNetB0
* Transfer Learning
* Fine-tuning
* Grad-CAM

### Data Science

* NumPy
* Pandas
* Scikit-learn

### Image Processing & Visualization

* OpenCV
* Pillow
* Matplotlib

---

## 📁 Repository Structure

```text
tumor-detection-mri/
│
├── data/
│
├── models/
│
├── initial report/
│
├── Base_CNN copy.ipynb
├── ResNet50.ipynb
├── efficientNet.ipynb
├── lr_ensemble.ipynb
├── modelsSummary.ipynb
│
├── app.py
├── compare.py
├── ensemble.py
├── evaluate.py
├── gradcam.py
│
├── loss_json/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Raya1313/tumor-detection-mri.git
cd tumor-detection-mri
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

The repository contains separate notebooks and Python scripts for different parts of the project.

### Model training

The model training experiments are contained in:

```text
Base_CNN copy.ipynb
ResNet50.ipynb
efficientNet.ipynb
```

### Model comparison

Use:

```text
compare.py
```

to compare model performance.

### Evaluation

The evaluation workflow is implemented in:

```text
evaluate.py
```

### Grad-CAM

Grad-CAM visualization is implemented in:

```text
gradcam.py
```

### Ensemble experiments

Ensemble experiments are contained in:

```text
ensemble.py
lr_ensemble.ipynb
```

### Model summary

Model architecture and parameter comparisons are explored in:

```text
modelsSummary.ipynb
```

---

## 🖥️ Application

The repository also contains an application entry point:

```text
app.py
```

This provides the basis for interacting with the trained models outside the training notebooks.

The application is intended as a demonstration of the trained classification models rather than a clinical diagnostic interface.

---

## ⚠️ Limitations

This project has several important limitations.

### Dataset Generalization

The models are primarily evaluated on the selected dataset. Performance may decrease substantially on external datasets.

### Dataset Bias

Models may learn characteristics specific to the dataset rather than clinically meaningful tumor features.

### 2D Classification

The project treats MRI images primarily as 2D images rather than using complete 3D MRI volumes.

### Class-Specific Performance

Overall accuracy does not fully describe model behavior. Some classes are more difficult to distinguish than others.

### Clinical Validation

The models have not undergone clinical validation and should not be considered medical diagnostic systems.

### Explainability

Grad-CAM highlights regions associated with model activation, but these visualizations do not prove that the highlighted regions correspond to clinically meaningful tumor features.

---

## 🔮 Future Work

Potential improvements include:

* Testing on independent external datasets.
* Evaluating cross-dataset generalization.
* Adding systematic data augmentation experiments.
* Performing more extensive hyperparameter optimization.
* Comparing additional pretrained architectures.
* Investigating patient-level dataset splitting.
* Exploring MRI segmentation before classification.
* Evaluating model calibration and prediction uncertainty.
* Improving ensemble strategies.
* Expanding explainability with additional XAI techniques.
* Developing a more complete interactive application.

---

## 📚 References

### Dataset

Masoud Nickparvar. **Brain Tumor MRI Dataset.** Kaggle.

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### ResNet

He, K., Zhang, X., Ren, S., & Sun, J.
**Deep Residual Learning for Image Recognition.**

### EfficientNet

Tan, M., & Le, Q. V.
**EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.**

### Grad-CAM

Selvaraju, R. R. et al.
**Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.**

---

## 👤 Author

**Raya**

GitHub: [@Raya1313](https://github.com/Raya1313)

---

## 📜 License

This repository currently does not specify a license.

If this project is intended for public reuse, consider adding an appropriate open-source license.

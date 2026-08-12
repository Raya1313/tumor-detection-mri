# 🧠 Brain Tumor MRI Classification

A deep learning project for **multi-class classification of brain MRI images** into four categories: **glioma, meningioma, pituitary tumor, and no tumor**.

The project compares a custom Convolutional Neural Network (CNN) with transfer-learning approaches using **ResNet50** and **EfficientNetB0**. It also explores model interpretability using **Grad-CAM** to visualize image regions that contribute to the model's predictions.

> **Disclaimer:** This project is intended for research and educational purposes only. It is not a medical diagnostic system and should not be used to make clinical decisions.

---

## 📌 Project Overview

Brain tumor classification from MRI scans is a challenging computer vision problem because tumors can vary considerably in size, shape, location, and appearance.

This project investigates whether deep learning models can effectively classify MRI scans into four tumor categories and compares the performance of different architectures.

### Classes

The models classify MRI scans into:

* **Glioma**
* **Meningioma**
* **Pituitary tumor**
* **No tumor**

---

## 🎯 Objectives

The main objectives of this project are to:

* Develop a baseline CNN for brain MRI classification.
* Compare the baseline model with pretrained **ResNet50** and **EfficientNetB0** architectures.
* Evaluate model performance using multiple classification metrics.
* Analyze class-specific performance using confusion matrices and classification reports.
* Investigate model predictions using **Grad-CAM**.
* Provide a foundation for further research into generalization and explainable medical image classification.

---

## 🗂️ Dataset

The project uses the **Brain Tumor MRI Dataset** from Kaggle.

The dataset contains MRI images belonging to four classes:

| Class      | Description                            |
| ---------- | -------------------------------------- |
| Glioma     | MRI scans containing glioma tumors     |
| Meningioma | MRI scans containing meningioma tumors |
| Pituitary  | MRI scans containing pituitary tumors  |
| No Tumor   | MRI scans without a detected tumor     |

Dataset source:

**Brain Tumor MRI Dataset**
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The images are resized to **224 × 224 pixels** for model training.

---

## 🏗️ Models

Three deep learning approaches are investigated.

### 1. Custom CNN

A convolutional neural network developed specifically for this project.

The baseline architecture consists of convolutional layers followed by pooling and fully connected layers for four-class classification.

The baseline model uses grayscale MRI images and pixel normalization.

### 2. ResNet50

A pretrained **ResNet50** model is used through transfer learning.

The original classification head is replaced with a custom head designed for the four MRI classes.

ResNet50 processes RGB images using the corresponding ResNet preprocessing pipeline.

### 3. EfficientNetB0

A pretrained **EfficientNetB0** model is also evaluated.

The model uses:

* ImageNet pretrained weights
* `224 × 224 × 3` input
* Global Average Pooling
* Dense classification layers
* Dropout regularization
* Four-class softmax output

Fine-tuning is used to adapt the pretrained model to the MRI classification task.

---

## 🔬 Experimental Pipeline

The general workflow is:

```text
MRI Dataset
     │
     ▼
Image Loading
     │
     ▼
Preprocessing & Resizing
     │
     ▼
Train / Validation / Test Sets
     │
     ├───────────────┬────────────────┐
     ▼               ▼                ▼
  Custom CNN      ResNet50       EfficientNetB0
     │               │                │
     └───────────────┴────────────────┘
                     │
                     ▼
              Model Evaluation
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Accuracy      Confusion     Classification
                  Matrix          Report
                     │
                     ▼
                 Grad-CAM
```

---

## 📊 Evaluation

Model performance is evaluated using more than accuracy alone.

The evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC-AUC where applicable
* Class-specific performance

This is particularly important for medical-image classification because overall accuracy can hide poor performance on individual classes.

---

## 🔍 Explainability with Grad-CAM

To investigate **why** the models make particular predictions, Grad-CAM is applied to selected convolutional layers.

Grad-CAM produces a heatmap showing the image regions that contributed most strongly to the model's prediction.

This allows the project to examine whether the models are focusing on visually meaningful regions of the MRI rather than irrelevant image features.

Target layers used for the models include:

```text
Custom CNN        → final convolutional layer
ResNet50          → conv5_block3_out
EfficientNetB0    → top_activation
```

Example workflow:

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
Gradient Calculation
   │
   ▼
Grad-CAM Heatmap
   │
   ▼
Visualization
```

---

## 📈 Results

The current experiments show that deep learning models can achieve strong classification performance on the selected dataset.

A representative evaluation has achieved approximately **92% test accuracy**, although performance varies between architectures and individual tumor classes.

Class-level analysis is also performed because some categories, particularly **glioma**, can be more difficult for the models to distinguish from other tumor types.

> Results should be interpreted in the context of the dataset and experimental setup. High performance on a single dataset does not necessarily imply reliable performance on unseen clinical data.

Detailed results, including confusion matrices, classification reports, and Grad-CAM visualizations, are included in the project notebooks/results where available.

---

## 🧪 Technologies

The project is implemented using Python and the following major libraries:

* Python
* TensorFlow
* Keras
* NumPy
* OpenCV
* Pillow
* Matplotlib
* Scikit-learn

### Deep Learning

* Convolutional Neural Networks
* Transfer Learning
* ResNet50
* EfficientNetB0
* Grad-CAM

---

## 📁 Project Structure

The repository is organized around model training, evaluation, and explainability.

A typical structure is:

```text
tumor-detection-mri/
│
├── models/
│   ├── BaseCNN/
│   ├── ResNet50/
│   └── EfficientNetB0/
│
├── notebooks/
│   ├── training/
│   ├── evaluation/
│   └── GradCAM/
│
├── results/
│   ├── confusion_matrices/
│   ├── classification_reports/
│   └── gradcam/
│
├── preprocessing/
│
├── requirements.txt
├── README.md
└── .gitignore
```

*The exact structure should be updated to match the repository files.*

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Raya1313/tumor-detection-mri.git
cd tumor-detection-mri
```

### 2. Create a virtual environment

```bash
python -m venv .venv
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

### 4. Download the dataset

Download the Brain Tumor MRI Dataset from Kaggle and place it according to the dataset structure expected by the notebooks/scripts.

### 5. Train a model

Run the corresponding training notebook or script.

### 6. Evaluate the model

Run the evaluation workflow to generate:

* Classification reports
* Confusion matrices
* Accuracy/loss curves
* Additional evaluation metrics

### 7. Generate Grad-CAM visualizations

Use the Grad-CAM workflow with a trained model and an MRI image to visualize the regions contributing to the prediction.

---

## ⚠️ Limitations

Several limitations should be considered:

1. **Dataset generalization**

   Performance on the selected dataset does not guarantee equivalent performance on MRI scans from different hospitals, scanners, populations, or acquisition protocols.

2. **Dataset bias**

   Deep learning models can learn dataset-specific characteristics that are unrelated to the underlying medical condition.

3. **Class imbalance**

   Differences in the number and characteristics of samples across classes can influence model performance.

4. **2D image classification**

   MRI data is inherently volumetric, while this project primarily treats individual images as 2D inputs.

5. **No clinical validation**

   The models have not undergone clinical validation and should not be considered diagnostic tools.

6. **Explainability limitations**

   Grad-CAM provides an indication of image regions associated with a prediction, but it does not prove that the highlighted region represents the actual pathological feature used by a clinician.

---

## 🔮 Future Work

Potential directions for improving the project include:

* Testing on completely independent MRI datasets.
* Applying data augmentation and evaluating its effect on generalization.
* Performing systematic hyperparameter optimization.
* Comparing additional pretrained architectures.
* Investigating patient-level rather than image-level splitting.
* Exploring MRI segmentation before classification.
* Evaluating calibration and uncertainty.
* Conducting external validation on datasets from different sources.
* Improving explainability using additional XAI techniques.
* Developing an interactive demonstration application.

---

## 📚 References

* Kaggle Brain Tumor MRI Dataset
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

* He, K., Zhang, X., Ren, S., & Sun, J. *Deep Residual Learning for Image Recognition.*

* Tan, M., & Le, Q. V. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*

* Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*

---

## 📜 License

Add the license appropriate for the source code and dataset usage before publishing the repository as a research project.

---

## 👤 Author

**Raya**

GitHub: https://github.com/Raya1313

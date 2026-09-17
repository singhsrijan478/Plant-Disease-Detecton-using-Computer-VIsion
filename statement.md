# Project Statement

## Project Title

**Plant Disease Detection Using Computer Vision**

---

## 1. Problem Statement

Plant diseases can negatively affect plant health and crop productivity. Identifying diseases from the appearance of leaves can also be difficult for people without specialized agricultural knowledge, particularly when symptoms are subtle or visually similar to healthy leaves.

This project aims to develop a computer vision-based system that can analyze an image of a plant leaf and classify it into one of three categories: **Healthy, Powdery, or Rust**.

The system provides a simple command-line interface through which a user can supply a leaf image and receive a predicted condition along with the model's confidence.

---

## 2. Project Scope

The scope of the project includes the development of an image classification pipeline covering:

* Plant leaf image preprocessing
* Training data augmentation
* Deep learning model development
* Transfer learning using MobileNetV2
* Plant disease classification
* Single-image prediction
* Prediction confidence calculation
* Model evaluation
* Classification report generation
* Confusion matrix generation
* Per-class accuracy analysis

The current system is limited to the three supported classes:

```text
Healthy
Powdery
Rust
```

The project focuses on **image-based classification** and does not currently provide disease treatment, agricultural diagnosis, disease severity estimation, or real-time field monitoring.

---

## 3. Target Users

The primary target users are:

* **Household plant owners** who want a quick preliminary indication of whether a plant leaf appears healthy or shows characteristics associated with Powdery or Rust.
* **Students and learners** interested in understanding the application of computer vision and deep learning to agricultural problems.
* **Developers and researchers** who want to experiment with image-based plant disease classification.

The system is intended to provide an accessible preliminary screening tool rather than replace professional agricultural expertise.

---

## 4. High-Level Features

### 4.1 Image Preprocessing

The system accepts a plant leaf image and prepares it for model inference by resizing it to the required dimensions and normalizing its pixel values.

### 4.2 Data Augmentation

During model training, image augmentation techniques such as rotation, shifting, flipping, and zooming are applied to increase variation in the training data.

### 4.3 Deep Learning Classification

A pretrained MobileNetV2 architecture is used as a feature extractor. A custom classification layer is added to classify images into the supported plant-condition categories.

### 4.4 Disease Prediction

The user can provide an individual leaf image through the command line. The system returns:

* Predicted class
* Confidence score

### 4.5 Model Evaluation

The system provides multiple evaluation outputs, including:

* Accuracy
* Precision
* Recall
* F1-score
* Per-class accuracy
* Confusion matrix

### 4.6 Error Analysis

The system's evaluation results can be used to identify misclassified images and analyze similarities between visually related classes.

In the current evaluation, a Powdery sample was misclassified as a visually similar class. Manual inspection indicated that the appearance of the sample was similar to a healthy leaf, demonstrating the challenge of distinguishing subtle disease symptoms using image classification.

---

## 5. Expected Outcome

The expected outcome is a modular command-line application capable of taking a plant leaf image as input and producing a classification among the supported categories.

The overall workflow is:

```text
Plant Leaf Image
       ↓
Image Preprocessing
       ↓
MobileNetV2 Feature Extraction
       ↓
Classification Layer
       ↓
Predicted Condition
       ↓
Confidence Score
```

The project also aims to demonstrate the complete machine-learning development workflow, from data preprocessing and model development to prediction, evaluation, and error analysis.

---

## 6. Project Limitations

The current implementation has the following limitations:

* Only three plant-condition classes are supported.
* The available evaluation subset contains 29 images.
* Visually similar Healthy and Powdery leaves can be difficult to distinguish.
* Prediction performance can be affected by image quality, lighting, background, leaf orientation, and other image conditions.
* The current evaluation subset is derived from the available dataset and is not a completely independent external test dataset.

---

## 7. Future Scope

Possible future extensions include:

* Expanding the dataset with more plant varieties and disease stages.
* Supporting additional plant diseases.
* Fine-tuning the pretrained MobileNetV2 layers.
* Improving leaf segmentation and image preprocessing.
* Detecting disease severity.
* Developing a graphical or web-based interface.
* Providing general information about detected plant conditions.
* Evaluating the system using a larger independent test dataset.

# Plant Disease Detection Using Computer Vision

A computer vision and deep learning project that identifies common plant leaf conditions from an image. The system uses a lightweight **MobileNetV2** model with transfer learning to classify leaf images into three categories: **Healthy, Powdery, and Rust**.

The project is designed as a simple command-line tool, allowing a user to provide a plant leaf image and receive a predicted condition along with the model's confidence.

---

## Project Overview

Plant diseases can be difficult to identify at an early stage, particularly for people who grow plants at home and may not have access to expert knowledge.

This project explores how computer vision can be used to provide a quick preliminary screening of plant leaves from photographs.

The system:

1. Accepts a plant leaf image.
2. Preprocesses the image to the required input format.
3. Passes the image through a trained MobileNetV2-based neural network.
4. Predicts the corresponding plant condition.
5. Displays the predicted class and confidence score.

The current model supports:

* **Healthy**
* **Powdery**
* **Rust**

> **Note:** This project is intended as an image-based screening tool and should not be considered a replacement for professional agricultural diagnosis.

---

## Features

* Image-based plant disease classification
* MobileNetV2 transfer-learning architecture
* CPU-compatible model execution
* Image preprocessing and normalization
* Data augmentation during training
* Command-line interface
* Prediction confidence reporting
* Classification report generation
* Confusion matrix generation
* Per-class accuracy calculation
* Reproducible evaluation using a fixed random seed
* Modular project architecture

---

## Technologies Used

| Technology         | Purpose                                |
| ------------------ | -------------------------------------- |
| Python             | Core programming language              |
| TensorFlow / Keras | Deep learning and model training       |
| MobileNetV2        | Transfer-learning feature extractor    |
| NumPy              | Numerical computation                  |
| OpenCV / PIL       | Image processing                       |
| Scikit-learn       | Model evaluation                       |
| Matplotlib         | Result visualization                   |
| Seaborn            | Confusion matrix visualization         |
| Git / GitHub       | Version control and project management |

---

## Model Architecture

The project uses **MobileNetV2** as the feature extraction backbone.

The pretrained ImageNet weights are used while the base MobileNetV2 layers remain frozen. A small classification head is added on top:

```text
Input Image
     │
     ▼
Resize to 224 × 224
     │
     ▼
Pixel Normalization
     │
     ▼
MobileNetV2
(Pretrained Feature Extractor)
     │
     ▼
Global Average Pooling
     │
     ▼
Dropout (0.3)
     │
     ▼
Dense Layer (64 neurons)
     │
     ▼
Softmax Output
     │
     ▼
Healthy / Powdery / Rust
```

### Why MobileNetV2?

MobileNetV2 was selected because it provides a relatively lightweight convolutional neural network architecture while still providing strong image feature extraction capabilities.

This is particularly useful for the project because the system is intended to run on a normal CPU without requiring a dedicated GPU.

---

## Dataset

The project uses a plant leaf image dataset containing three classes:

```text
Healthy
Powdery
Rust
```

The images are organized into class-specific directories so that they can be loaded using Keras' directory-based image generator.

During training, data augmentation is applied to increase variation in the training images. The augmentation includes:

* Rotation
* Width and height shifting
* Horizontal flipping
* Zooming

Validation/evaluation images are **not augmented**. They are only rescaled so that evaluation remains deterministic.

---

## Image Preprocessing

Input images are processed before being passed to the neural network.

The preprocessing pipeline includes:

1. Loading the image.
2. Converting it into a compatible image representation.
3. Resizing it to **224 × 224 pixels**.
4. Converting pixel values to floating-point values.
5. Normalizing pixel values to the range **0–1**.
6. Adding the batch dimension required by the neural network.

---

## Project Structure

```text
plant_disease_project/
│
├── data/
│   ├── README.md
│   └── plant-disease-recognition-dataset/
│       └── Test/
│           ├── Healthy/
│           ├── Powdery/
│           └── Rust/
│
├── model/
│   └── plant_disease_model.keras
│
├── results/
│   ├── confusion_matrix.png
│   └── training_history.png
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── prediction.py
│   └── evaluation.py
│
├── tests/
│   └── __init__.py
│
├── test_images/
│   └── 8def3f60308ab41b.jpg
│
├── .gitignore
├── requirements.txt
├── statement.md
├── README.md
└── plant_disease_detector_cpu.py
```

### Module Description

| Module             | Responsibility                                        |
| ------------------ | ----------------------------------------------------- |
| `main.py`          | Command-line interface and application control        |
| `model.py`         | MobileNetV2 model construction and compilation        |
| `preprocessing.py` | Dataset and image preprocessing                       |
| `prediction.py`    | Single-image prediction and result display            |
| `evaluation.py`    | Classification report and confusion matrix generation |
| `tests/`           | Automated testing components                          |
| `model/`           | Stores the trained model                              |
| `results/`         | Stores generated evaluation visualizations            |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/singhsrijan478/Plant-Disease-Detecton-using-Computer-VIsion.git
```

### 2. Enter the project directory

```bash
cd Plant-Disease-Detecton-using-Computer-VIsion
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

It is recommended to use a Python virtual environment.

Example:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Then install the requirements:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The project can be executed directly from the command line.

### Predict a Plant Disease

Provide the path to an image using `--predict`:

```bash
python src\main.py --predict test_images\8def3f60308ab41b.jpg
```

Example output:

```text
========================================
       PLANT DISEASE DETECTION
========================================

Image      : test_images\8def3f60308ab41b.jpg
Prediction : Rust
Confidence : 93.00%

========================================
```

The confidence value represents the model's confidence for that particular prediction. It is **not the overall accuracy of the model**.

---

## Model Evaluation

To evaluate the model:

```bash
python src\main.py --eval
```

The evaluation process generates:

* Classification report
* Precision
* Recall
* F1-score
* Per-class accuracy
* Overall accuracy
* Confusion matrix

The confusion matrix is automatically saved to:

```text
results/confusion_matrix.png
```

---

## Evaluation Results

The current evaluation was performed on **29 evaluation images**.

The model produced:

```text
Correct predictions   : 28
Incorrect predictions : 1
Total samples         : 29

Overall Accuracy      : 96.55%
```

### Per-Class Accuracy

| Class   | Accuracy |
| ------- | -------: |
| Healthy |  100.00% |
| Powdery |   90.00% |
| Rust    |  100.00% |

The model therefore correctly classified **28 out of 29 evaluation images**.

---

## Confusion Matrix

The generated confusion matrix can be found at:

```text
results/confusion_matrix.png
```

The confusion matrix provides a class-by-class view of the model's predictions and makes it possible to identify which categories are being confused with one another.

---

## Analysis of the Misclassification

One Powdery sample was incorrectly classified.

The image was manually inspected and showed strong visual similarity to the Healthy class. The leaf did not contain highly distinctive visual symptoms, making the distinction between the two classes difficult.

This highlights an important challenge in plant disease classification: **some disease symptoms can visually overlap with healthy plant characteristics**, particularly when symptoms are subtle.

The misclassification therefore provides useful information about a limitation of the current model rather than simply representing an isolated numerical error.

Potential improvements include:

* Increasing the number of training images.
* Including more variation in leaf appearance.
* Including different stages of disease development.
* Adding higher-quality and more diverse images.
* Fine-tuning additional MobileNetV2 layers.
* Exploring more advanced image preprocessing techniques.

---

## Command-Line Options

The main application supports several operations:

### Train the model

```bash
python src\main.py --train
```

Specify the number of epochs:

```bash
python src\main.py --train --epochs 10
```

### Predict an image

```bash
python src\main.py --predict path\to\image.jpg
```

### Evaluate the model

```bash
python src\main.py --eval
```

### Generate the confusion matrix

```bash
python src\main.py --confusion
```

### Specify a custom dataset directory

```bash
python src\main.py --eval --data_dir path\to\dataset
```

---

## Functional Modules

The project is divided into multiple functional modules.

### 1. Data Preprocessing

**Input:** Raw plant images

**Process:** Image loading, resizing, normalization, and training augmentation

**Output:** Processed image data suitable for model training/evaluation

### 2. Model Construction

**Input:** Number of plant disease classes

**Process:** MobileNetV2 feature extraction and classification-head construction

**Output:** Compiled deep learning model

### 3. Disease Prediction

**Input:** Individual plant leaf image

**Process:** Preprocessing followed by neural-network inference

**Output:** Predicted disease class and confidence

### 4. Model Evaluation

**Input:** Evaluation dataset and trained model

**Process:** Generate predictions and calculate evaluation metrics

**Output:** Classification report, per-class accuracy, overall accuracy, and confusion matrix

---

## Non-Functional Requirements

The system is designed with the following non-functional requirements:

### Performance

The lightweight MobileNetV2 architecture allows inference to be performed on CPU-based systems.

### Usability

The application can be operated using simple command-line commands without requiring a graphical interface.

### Reliability

Input validation is performed before image processing, and missing image/model files produce clear error messages.

### Maintainability

The system is divided into independent modules for preprocessing, model construction, prediction, and evaluation.

### Reproducibility

A fixed random seed is used during data preparation and model-related operations to improve reproducibility.

### Resource Efficiency

MobileNetV2 is used instead of a substantially larger convolutional architecture, reducing computational requirements.

---

## Limitations

The current system has several limitations:

* The model supports only three classes: Healthy, Powdery, and Rust.
* The evaluation set contains only 29 images.
* The current evaluation subset is derived from the available dataset rather than being a completely independent external test dataset.
* Images with subtle disease symptoms can be difficult to distinguish from healthy leaves.
* Prediction quality may be affected by lighting, camera quality, background clutter, leaf orientation, and image composition.
* A high confidence score does not guarantee that a prediction is correct.

---

## Future Enhancements

Future versions of the project could include:

1. **Larger Dataset**
   Train on a larger and more diverse collection of plant images.

2. **More Disease Classes**
   Extend the system to recognize additional plant diseases.

3. **Disease Severity Detection**
   Estimate whether a detected disease is at an early, moderate, or advanced stage.

4. **Model Fine-Tuning**
   Unfreeze selected MobileNetV2 layers and fine-tune them using plant-specific images.

5. **Improved Image Processing**
   Investigate background removal, leaf segmentation, and additional image enhancement techniques.

6. **User-Friendly Interface**
   Develop a web or desktop interface where users can upload a leaf photograph without using command-line commands.

7. **Treatment Recommendations**
   Integrate a knowledge base containing general information about detected conditions and possible management approaches.

---

## Project Objective

The primary objective of this project is to demonstrate how computer vision and transfer learning can be applied to a practical agricultural problem.

The system provides a simple pipeline from:

```text
Plant Leaf Image
       ↓
Image Preprocessing
       ↓
Deep Learning Model
       ↓
Disease Classification
       ↓
Prediction + Confidence
```

The project also demonstrates the complete machine-learning workflow, including data preprocessing, model development, prediction, evaluation, visualization, and modular software organization.

---

## Disclaimer

This project is intended for educational and preliminary screening purposes. Predictions should not be treated as a definitive agricultural or plant-health diagnosis. For important crops or uncertain cases, the result should be verified using appropriate agricultural expertise.

---

## Author

**Srijan Singh**

Plant Disease Detection using Computer Vision
VIT Bhopal University

---

## License

This project is intended for educational and academic purposes.

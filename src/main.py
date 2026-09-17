import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import build_model
from preprocessing import load_and_preprocess_data
from prediction import predict_image, display_prediction
from evaluation import (
    plot_confusion_matrix,
    generate_detailed_report
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "model/plant_disease_model.keras"
BEST_MODEL_PATH = "model/best_plant_model.keras"

DEFAULT_DATA_DIR = (
    "data/plant-disease-recognition-dataset/Test"
)

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def load_model_if_available():
    """Load the trained model if it exists."""

    if not os.path.exists(MODEL_PATH):
        print(
            "Error: Model not found. "
            "Please train the model first using --train."
        )
        return None

    return keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_model(data_dir, epochs):
    """Train and save the plant disease detection model."""

    print("Starting model training on CPU...")

    train_gen, val_gen, _ = load_and_preprocess_data(
        data_dir
    )

    class_names = list(
        train_gen.class_indices.keys()
    )

    print(
        f"Classes detected: {class_names}"
    )

    model = build_model(
        len(class_names)
    )

    os.makedirs("model", exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=max(
            1,
            train_gen.n // train_gen.batch_size
        ),
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=max(
            1,
            val_gen.n // val_gen.batch_size
        ),
        callbacks=callbacks,
        verbose=1
    )

    model.save(MODEL_PATH)

    print("\nModel trained and saved successfully!")
    print(f"Model location: {MODEL_PATH}")

    return history


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def run_prediction(image_path, data_dir):
    """Run disease prediction on a single image."""

    print("Making prediction on image...")

    model = load_model_if_available()

    if model is None:
        return

    # Get class names from the dataset
    train_gen, _, _ = load_and_preprocess_data(
        data_dir
    )

    class_names = list(
        train_gen.class_indices.keys()
    )

    predicted_class, confidence = predict_image(
        model,
        image_path,
        class_names
    )

    display_prediction(
        image_path,
        predicted_class,
        confidence
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def run_evaluation(data_dir):
    """Run complete model evaluation."""

    print("Running comprehensive evaluation...")

    model = load_model_if_available()

    if model is None:
        return

    _, _, test_gen = load_and_preprocess_data(
        data_dir
    )

    class_names = list(
        test_gen.class_indices.keys()
    )

    generate_detailed_report(
        model,
        test_gen,
        class_names
    )


def run_confusion_matrix(data_dir):
    """Generate only the confusion matrix."""

    print("Generating confusion matrix...")

    model = load_model_if_available()

    if model is None:
        return

    _, _, test_gen = load_and_preprocess_data(
        data_dir
    )

    class_names = list(
        test_gen.class_indices.keys()
    )

    plot_confusion_matrix(
        model,
        test_gen,
        class_names
    )


# --------------------------------------------------
# Main CLI
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plant Disease Detection "
            "(CPU Optimized)"
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to the dataset directory"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model"
    )

    parser.add_argument(
        "--predict",
        type=str,
        help="Path to image for prediction"
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate model"
    )

    parser.add_argument(
        "--confusion",
        action="store_true",
        help="Generate confusion matrix"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    args = parser.parse_args()

    if args.train:

        train_model(
            args.data_dir,
            args.epochs
        )

    elif args.predict:

        run_prediction(
            args.predict,
            args.data_dir
        )

    elif args.eval:

        run_evaluation(
            args.data_dir
        )

    elif args.confusion:

        run_confusion_matrix(
            args.data_dir
        )

    else:

        parser.print_help()


if __name__ == "__main__":
    main()
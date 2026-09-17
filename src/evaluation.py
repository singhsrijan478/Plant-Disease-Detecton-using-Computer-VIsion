import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# Project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Results directory
RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)


def get_predictions(model, test_generator):
    """
    Generate predictions exactly once.

    The returned values are reused for the classification
    report and confusion matrix.
    """

    print("\nGenerating predictions...")

    test_generator.reset()

    predictions = model.predict(
        test_generator,
        verbose=1
    )

    y_pred_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true = test_generator.classes

    print(
        f"\nSamples evaluated: {len(y_true)}"
    )

    return y_true, y_pred_classes


def plot_confusion_matrix(
    y_true,
    y_pred_classes,
    class_names
):
    """
    Generate and save the confusion matrix.
    """

    print("\nGenerating confusion matrix...")

    cm = confusion_matrix(
        y_true,
        y_pred_classes
    )

    total_samples = np.sum(cm)

    correct_predictions = np.trace(cm)

    accuracy = (
        correct_predictions / total_samples
        if total_samples > 0
        else 0
    )

    # Make sure the results directory exists.
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print(
        f"Saving results to: {RESULTS_DIR}"
    )

    # ---------------------------------------------------------
    # Create confusion matrix
    # ---------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={
            "label": "Number of Samples"
        }
    )

    plt.title(
        "Confusion Matrix - Plant Disease Detection",
        fontsize=16,
        fontweight="bold"
    )

    plt.ylabel(
        "True Label",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(
        "Predicted Label",
        fontsize=14,
        fontweight="bold"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.figtext(
        0.15,
        0.02,
        f"Overall Accuracy: {accuracy:.2%}",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            facecolor="lightgray",
            alpha=0.7
        )
    )

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save confusion matrix
    # ---------------------------------------------------------

    output_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    print(
        f"Saving confusion matrix to: {output_path}"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Confusion matrix saved successfully!"
    )

    return cm, accuracy


def generate_detailed_report(
    model,
    test_generator,
    class_names
):
    """
    Generate a complete model evaluation report.

    All metrics use exactly the same predictions.
    """

    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Generate predictions ONCE
    # ---------------------------------------------------------

    y_true, y_pred_classes = get_predictions(
        model,
        test_generator
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    print("\nClassification Report:")
    print("=" * 40)

    report = classification_report(
        y_true,
        y_pred_classes,
        target_names=class_names,
        digits=2
    )

    print(report)

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm, accuracy = plot_confusion_matrix(
        y_true,
        y_pred_classes,
        class_names
    )

    # ---------------------------------------------------------
    # Per-class accuracy
    # ---------------------------------------------------------

    print("\nPer-Class Accuracy:")
    print("=" * 40)

    class_accuracy = []

    for i, class_name in enumerate(class_names):

        total_class_samples = cm[i].sum()

        if total_class_samples > 0:
            class_acc = (
                cm[i, i] /
                total_class_samples
            )
        else:
            class_acc = 0

        class_accuracy.append(
            class_acc
        )

        print(
            f"{class_name:<15}: "
            f"{class_acc:.2%}"
        )

    # ---------------------------------------------------------
    # Overall evaluation summary
    # ---------------------------------------------------------

    correct = np.trace(cm)

    total = np.sum(cm)

    incorrect = total - correct

    print("\nEvaluation Summary:")
    print("=" * 40)

    print(
        f"Correct predictions  : {correct}"
    )

    print(
        f"Incorrect predictions: {incorrect}"
    )

    print(
        f"Total samples        : {total}"
    )

    print(
        f"Overall Accuracy     : {accuracy:.2%}"
    )

    print("=" * 60)

    return cm, accuracy
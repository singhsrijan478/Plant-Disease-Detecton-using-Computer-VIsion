import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42


def load_and_preprocess_data(base_dir):
    """
    Load and preprocess the plant disease dataset.

    Training data uses augmentation.
    Validation data uses rescaling only to ensure
    deterministic and unbiased evaluation.
    """

    print("Loading data from:", base_dir)

    # ---------------------------------------------------------
    # Training data augmentation
    # ---------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode="nearest",
        validation_split=0.2
    )

    # ---------------------------------------------------------
    # Validation data: NO augmentation
    # ---------------------------------------------------------
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    # ---------------------------------------------------------
    # Training generator
    # ---------------------------------------------------------
    print("Creating training generator...")

    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        subset="training",
        seed=SEED
    )

    # ---------------------------------------------------------
    # Validation generator
    # ---------------------------------------------------------
    print("Creating validation generator...")

    validation_generator = validation_datagen.flow_from_directory(
        base_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        subset="validation",
        seed=SEED
    )

    # The current project uses the validation subset for evaluation.
    # It is kept as test_generator for compatibility with the
    # existing main.py structure.
    test_generator = validation_generator

    print("\nDataset information:")
    print(f"Training samples   : {train_generator.samples}")
    print(f"Validation samples : {validation_generator.samples}")
    print(f"Total samples      : "
          f"{train_generator.samples + validation_generator.samples}")

    print("\nClass mapping:")
    print(train_generator.class_indices)

    return train_generator, validation_generator, test_generator


def preprocess_image(image_path):
    """
    Load and preprocess a single image for prediction.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array
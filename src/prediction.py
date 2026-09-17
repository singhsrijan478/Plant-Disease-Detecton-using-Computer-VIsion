import numpy as np

from preprocessing import preprocess_image


def predict_image(model, image_path, class_names):
    """Predict the disease class of a plant leaf image."""

    image_array = preprocess_image(image_path)

    predictions = model.predict(image_array, verbose=0)

    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    predicted_class = class_names[predicted_class_idx]

    return predicted_class, confidence


def display_prediction(image_path, predicted_class, confidence):
    """Display prediction results in the CLI."""

    print("\n" + "=" * 40)
    print("       PLANT DISEASE DETECTION")
    print("=" * 40)

    print(f"\nImage      : {image_path}")
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2%}")

    print("\n" + "=" * 40)
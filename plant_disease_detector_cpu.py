# plant_disease_detector_cpu.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from PIL import Image
import argparse

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configure TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Using CPU for computation")

def load_and_preprocess_data(base_dir):
    """Load and preprocess the plant disease dataset"""
    print("Loading data from:", base_dir)
    
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for test set
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    print("Creating training generator...")
    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )
    
    print("Creating validation generator...")
    validation_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )
    
    # For this dataset, we'll use the validation set for testing
    test_generator = validation_generator
    
    return train_generator, validation_generator, test_generator

def build_model(num_classes):
    """Build the model"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg',
        alpha=0.35
    )
    
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_confusion_matrix(model, test_generator, class_names):
    """Plot and save confusion matrix"""
    print("Generating confusion matrix...")
    
    # Get predictions
    test_generator.reset()
    y_pred = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Confusion Matrix - Plant Disease Detection', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy information
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.15, 0.02, f'Overall Accuracy: {accuracy:.2%}', 
                fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm, accuracy

def generate_detailed_report(model, test_generator, class_names):
    """Generate comprehensive evaluation report"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Get predictions
    test_generator.reset()
    y_pred = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Classification report
    print("\n📊 Classification Report:")
    print("="*40)
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(report)
    
    # Confusion matrix
    cm, accuracy = plot_confusion_matrix(model, test_generator, class_names)
    
    # Per-class accuracy
    print("\n🎯 Per-Class Accuracy:")
    print("="*40)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15}: {class_accuracy[i]:.2%}")
    
    print(f"\n✅ Overall Accuracy: {accuracy:.2%}")
    
    return cm, accuracy

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plant Disease Detection (CPU Optimized)')
    parser.add_argument('--data_dir', type=str, default='data/plant-disease-recognition-dataset/Test',
                       help='Path to the dataset directory')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Path to image for prediction')
    parser.add_argument('--eval', action='store_true', help='Evaluate model and show detailed report')
    parser.add_argument('--confusion', action='store_true', help='Generate confusion matrix only')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting model training on CPU...")
        train_gen, val_gen, test_gen = load_and_preprocess_data(args.data_dir)
        class_names = list(train_gen.class_indices.keys())
        
        print(f"Classes detected: {class_names}")
        
        model = build_model(len(class_names))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
            ModelCheckpoint('best_plant_model.keras', monitor='val_accuracy', 
                           save_best_only=True, mode='max', verbose=1)
        ]
        
        # Train model
        history = model.fit(
            train_gen,
            steps_per_epoch=max(1, train_gen.n // train_gen.batch_size),
            epochs=args.epochs,
            validation_data=val_gen,
            validation_steps=max(1, val_gen.n // val_gen.batch_size),
            callbacks=callbacks,
            verbose=1
        )
        
        model.save('plant_disease_model.keras')
        print("Model trained and saved successfully!")
        
    elif args.eval:
        print("Running comprehensive evaluation...")
        if os.path.exists('plant_disease_model.keras'):
            model = keras.models.load_model('plant_disease_model.keras')
            _, _, test_gen = load_and_preprocess_data(args.data_dir)
            class_names = list(test_gen.class_indices.keys())
            
            # Generate detailed report with confusion matrix
            generate_detailed_report(model, test_gen, class_names)
            
        else:
            print("Error: Model not found. Please train the model first using --train")
    
    elif args.confusion:
        print("Generating confusion matrix...")
        if os.path.exists('plant_disease_model.keras'):
            model = keras.models.load_model('plant_disease_model.keras')
            _, _, test_gen = load_and_preprocess_data(args.data_dir)
            class_names = list(test_gen.class_indices.keys())
            
            # Generate only confusion matrix
            plot_confusion_matrix(model, test_gen, class_names)
            
        else:
            print("Error: Model not found. Please train the model first using --train")
    
    elif args.predict:
        print("Making prediction on image...")
        if os.path.exists('plant_disease_model.keras'):
            model = keras.models.load_model('plant_disease_model.keras')
            train_gen, _, _ = load_and_preprocess_data(args.data_dir)
            class_names = list(train_gen.class_indices.keys())
            
            # Prediction code here (same as before)
            img = Image.open(args.predict).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            print(f"\nPrediction Results:")
            print(f"Class: {class_names[predicted_class_idx]}")
            print(f"Confidence: {confidence:.2%}")
            
        else:
            print("Error: Model not found. Please train the model first using --train")
    
    else:
        print("Please specify an action: --train, --predict, --eval, or --confusion")

if __name__ == "__main__":
    main()

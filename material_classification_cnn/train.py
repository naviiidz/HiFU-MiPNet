"""
Author: Navid Zarrabi (Navid.Zarrabi@torontomu.ca)
Date: May 26, 2025
Purpose: Train a CNN model for material identification of microspheres.

Key Notes:
1. Enhanced signal processing for improved material discrimination.
2. Balanced dataset creation with equal samples per class.
3. Proper train/test split to prevent data leakage.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from preprocessing import process_signal, prepare_balanced_dataset
from model import create_model
from visualize import plot_training_history, plot_confusion_matrix
from config import get_training_config, preprocessing_parameters
import os

# Load preprocessing parameters
params = preprocessing_parameters()
SAMPLING_RATE = params["sampling_rate"]
LOWCUT_FREQ = params["lowcut_freq"]
HIGHCUT_FREQ = params["highcut_freq"]
SIGNAL_LENGTH = params["signal_length"]
SAMPLES_PER_CLASS = params["samples_per_class"]


def main():
    """Train the CNN model for material identification."""
    # File paths
    # Get the directory of the current script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(root_dir)

    csv_file = os.path.join(parent_dir, "data/") + "train_test_with_pmma.csv"
    model_path = os.path.join(root_dir, "final_model.h5")
    encoder_path = os.path.join(parent_dir, "data/") + "label_encoder_classes.npy"

    # Verify paths
    print("\nVerifying paths...")
    print(f"Root directory exists: {os.path.exists(root_dir)}")
    print(f"CSV file exists: {os.path.exists(csv_file)}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        # Load and prepare data
        print("\nLoading data...")
        data = pd.read_csv(csv_file)
        print(f"Loaded {len(data)} rows from CSV")
        print("Columns:", data.columns.tolist())

        print("\nConverting signals...")
        raw_signals = []
        signal_lengths = []
        for i, signal_str in enumerate(data['Signal'].values):
            try:
                signal = np.fromstring(signal_str, sep=',')
                raw_signals.append(signal)
                signal_lengths.append(len(signal))
            except Exception as e:
                print(f"Error converting signal {i}: {str(e)}")
                print(f"Signal string preview: {signal_str[:100]}...")
                continue

        print("\nSignal length statistics:")
        print(f"Min length: {min(signal_lengths)}")
        print(f"Max length: {max(signal_lengths)}")
        print(f"Mean length: {np.mean(signal_lengths):.2f}")

        material_types = data['Material Type'].values
        print(f"\nUnique material types: {np.unique(material_types)}")

        print("\nProcessing signals...")
        processed_signals = []
        for i, signal in enumerate(raw_signals):
            try:
                processed = process_signal(signal)
                processed_signals.append(processed)
                if i % 1000 == 0:
                    print(f"Processed {i}/{len(raw_signals)} signals")
            except Exception as e:
                print(f"Error processing signal {i}: {str(e)}")
                continue

        if not processed_signals:
            raise ValueError("No signals were successfully processed")

        print(f"\nTotal processed signals: {len(processed_signals)}")

        print("\nCreating balanced dataset...")
        balanced_signals, balanced_labels = prepare_balanced_dataset(
            processed_signals, material_types
        )

        print("\nSplitting data...")
        train_signals, test_signals, train_labels, test_labels = train_test_split(
            balanced_signals, balanced_labels,
            test_size=0.2, random_state=50,
            stratify=balanced_labels
        )

        print("\nPreparing signals for training...")
        padded_train = pad_sequences(train_signals, maxlen=SIGNAL_LENGTH,
                                   dtype='float32', padding='post', truncating='post')
        padded_test = pad_sequences(test_signals, maxlen=SIGNAL_LENGTH,
                                  dtype='float32', padding='post', truncating='post')

        padded_train = np.expand_dims(padded_train, axis=2)
        padded_test = np.expand_dims(padded_test, axis=2)

        print(f"Training data shape: {padded_train.shape}")
        print(f"Testing data shape: {padded_test.shape}")

        # Encode labels
        label_encoder = LabelEncoder()
        train_labels_encoded = label_encoder.fit_transform(train_labels)
        test_labels_encoded = label_encoder.transform(test_labels)

        categorical_train = to_categorical(train_labels_encoded)
        categorical_test = to_categorical(test_labels_encoded)

        # Save label encoder classes
        np.save(encoder_path, label_encoder.classes_)
        print(f"\nLabel encoder classes saved to {encoder_path}")

        # Create and train model
        print("\nCreating model...")
        model = create_model(
            input_shape=(SIGNAL_LENGTH, 1),
            num_classes=len(label_encoder.classes_)
        )

        # Get training configuration
        training_config = get_training_config()

        print("\nTraining model...")
        history = model.fit(
            padded_train,
            categorical_train,
            validation_data=(padded_test, categorical_test),
            **training_config
        )

        # Save model
        model.save(model_path)
        print(f"\nModel saved to {model_path}")

        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(padded_test, categorical_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Generate predictions
        test_predictions = model.predict(padded_test)
        predicted_labels = np.argmax(test_predictions, axis=1)

        # Plot results
        plot_training_history(history)
        plot_confusion_matrix(test_labels_encoded, predicted_labels,
                            label_encoder.classes_)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(test_labels_encoded, predicted_labels,
                                 target_names=label_encoder.classes_,
                                 digits=4))

    except Exception as e:
        print("\nError during training:")
        print(str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
Navid.Zarrabi@torontomu.ca
October 25, 2024
Goal: Testing a network using stored model for material identification
'''

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
from config import get_training_config, preprocessing_parameters, predict_config, peak_extract_param
from peak_extract import find_prominent_maxima, get_dataset_params
import h5py
import os


import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches



# Configuration
# Load preprocessing parameters
params = preprocessing_parameters()
SAMPLING_RATE = params["sampling_rate"]
LOWCUT_FREQ = params["lowcut_freq"]
HIGHCUT_FREQ = params["highcut_freq"]
SIGNAL_LENGTH = params["signal_length"]
SAMPLES_PER_CLASS = params["samples_per_class"]



def plot_results(max_amplitude_array, maxima_coords, decoded_labels, category=None, file_name=None):
    """Plot the results with material identification boxes and labels."""
   
    
    # Get box size from file-specific parameters
    params = get_dataset_params(category, file_name)
    box_size = params["box_size"]

    plt.figure(figsize=(6, 6))
    sns.heatmap(max_amplitude_array, cmap='viridis', cbar=True)
    plt.title("Material Identification Results")
    plt.xlabel("Y Index")
    plt.ylabel("X Index")

    ax = plt.gca()

    for coord, label_ in zip(maxima_coords, decoded_labels):
        # Add bounding box
        bbox = patches.Rectangle(
            (coord[1] - box_size / 2, coord[0] - box_size / 2),
            box_size, box_size,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(bbox)

        # Add label
        text_x = coord[1]
        text_y = coord[0] - box_size / 2 - 2
        plt.text(text_x, text_y, label_, color='white', fontsize=12,
                ha='center', va='bottom')

    plt.show()

def predict_materials(mat_file_path, model_path, encoder_path, category=None, file_name=None):
    """Predict material types from a .mat file using a trained CNN model."""
    try:
        # Load the data
        with h5py.File(mat_file_path, 'r') as file:
            RF = np.array(file['RFdata']['RF']).transpose()

        # Compute max amplitude array
        max_amplitude_array = np.max(RF, axis=0)

        # Find prominent maxima using file-specific parameters
        maxima_coords, maxima_values = find_prominent_maxima(max_amplitude_array, category, file_name)

        # Load model and label encoder
        model = tf.keras.models.load_model(model_path)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(encoder_path)

        # Process signals at maxima locations
        maxima_signals = []
        for coord in maxima_coords:
            signal = RF[:, coord[0], coord[1]]
            processed_signal = process_signal(signal)
            maxima_signals.append(processed_signal)

        # Prepare signals for prediction
        padded_signals = pad_sequences(maxima_signals, maxlen=3271,
                                     dtype='float32', padding='post',
                                     truncating='post')
        padded_signals = np.expand_dims(padded_signals, axis=2)

        # Make predictions
        predictions = model.predict(padded_signals)
        predicted_labels = np.argmax(predictions, axis=1)
        decoded_labels = label_encoder.inverse_transform(predicted_labels)

        # Plot results with file-specific parameters
        plot_results(max_amplitude_array, maxima_coords, decoded_labels, category, file_name)

        # Print predictions with confidence
        print("\nPrediction Results:")
        print("------------------")
        print(f"Found {len(decoded_labels)} microspheres")
        for i, (label, coord) in enumerate(zip(decoded_labels, maxima_coords)):
            confidence = np.max(predictions[i]) * 100
            print(f"Microsphere {i+1}: {label} at position ({coord[0]}, {coord[1]}) "
                  f"with {confidence:.1f}% confidence")

        return decoded_labels, maxima_coords, predictions

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None, None

# Define available datasets by category
DATASETS = peak_extract_param()


def get_dataset_path(category, file_name, root_dir):
    """Get the full path for a dataset file."""
    try:
        if category not in DATASETS:
            raise ValueError(f"Category '{category}' not found in available datasets")

        if file_name not in DATASETS[category]['files']:
            raise ValueError(f"File '{file_name}' not found in category '{category}'")

        return os.path.join(root_dir, DATASETS[category]['files'][file_name]['path'])

    except Exception as e:
        print(f"Error getting dataset path: {str(e)}")
        return None

def list_available_datasets():
    """Print available datasets and their files with parameters."""
    print("\nAvailable Datasets:")
    print("==================")
    for category, data in DATASETS.items():
        print(f"\n{category}")
        print("-" * len(category))
        print(f"Description: {data['description']}")
        print("\nFiles:")
        for name, file_info in data['files'].items():
            print(f"\n  - {name}:")
            print(f"    Path: {file_info['path']}")
            print("    Parameters:")
            for param_name, param_value in file_info['params'].items():
                print(f"      {param_name}: {param_value}")

def main():
    """Example usage of the prediction function."""
    # File paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(root_dir)
    model_path = os.path.join(root_dir, "final_model.h5")
    encoder_path = os.path.join(root_dir, "label_encoder_classes.npy")
    root_dir=os.path.join(parent_dir, "data")

    # List available datasets with their parameters
    list_available_datasets()

    # Example: predict materials in a steel sample
    category = "2024_04_17_MIXED"
    file_name = "glass_50um_small"

    print(f"\nProcessing {category}/{file_name}...")
    mat_file = get_dataset_path(category, file_name, root_dir)

    if mat_file:
        print("Processing file:", mat_file)
        # Pass both category and file_name to use file-specific parameters
        predict_materials(mat_file, model_path, encoder_path, category, file_name)

if __name__ == "__main__":
    main()
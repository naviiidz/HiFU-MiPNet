'''
Navid.Zarrabi@torontomu.ca
October 25, 2024
Goal: Getting strongest signals from microspheres and their neighbors, and generating labels for them including their location, type, and size
Notes: 
1. Use local maxima with sliding window
2. The code is semi-automatic, it uses some prior info and suggests a material and size, user has to supervise the label using optical images 
'''

import numpy as np
import h5py
from scipy.stats import skew, kurtosis
from scipy.fft import fft

from scipy.ndimage import gaussian_filter, maximum_filter
import numpy as np
from size_pred_labeling import train_and_predict_amplitude_classifier, calculate_thresholds, predict_amplitude_labels
from load_data import get_dataset_path, list_available_datasets
from plot import show_3d_max, plot_peaks, plot_heatmap

import csv
import sys
import os
import seaborn
import argparse

# Add the parent directory to load prediction modules and parameters
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'material_classification_cnn')))
from peak_extract import find_prominent_maxima, get_dataset_params
from peak_extract import find_prominent_maxima, get_dataset_params


size_labels=[80, 20]

default_material="pmma"
x_drop_pixels=0
y_drop_pixels=0


def is_signal(signal):
    """Extract various statistical features from a signal."""
    features = {}
    mean_val = np.mean(signal)

    features['mean'] = mean_val
    features['var'] = np.var(signal)
    features['skew'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)

    if features is None:
        return None
    return True


def main():
    # List available datasets with their parameters
    list_available_datasets()

    # # Step 1: Parse arguments
    parser = argparse.ArgumentParser(description="Predict materials in a steel sample.")
    parser.add_argument("--category", type=str, required=True, help="Category of the sample")
    parser.add_argument("--file_name", type=str, required=True, help="File name of the sample")
    parser.add_argument("--csv_output", type=str, default="output_labels.csv", help="Output CSV file name")
    args = parser.parse_args()

    # Step 2: Use parsed arguments
    category = args.category
    file_name = args.file_name
    csv_output = args.csv_output
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(root_dir)
    directory = get_dataset_path(category, file_name, os.path.join(parent_dir,"data/"))


    with h5py.File(directory, 'r') as file:
        RF = np.array(file['RFdata']['RF']).transpose()
        RFtime = np.array(file['RFdata']['RFtime']).transpose()

    # Compute envelopes for all signals
    signal_array_size=(RF.shape[0], RF.shape[1]-x_drop_pixels, RF.shape[2]-y_drop_pixels)
    signal_array = np.zeros(signal_array_size)

    for x_inst in range(signal_array_size[1]):
        for y_inst in range(signal_array_size[2]):
            x = RF[:, x_inst, y_inst]
            if is_signal(x):
                signal_array[:, x_inst, y_inst] += x

    max_amplitude_array = np.max(signal_array, axis=0)

    # display 3d plot to double check extracted peaks
    show_3d_max(max_amplitude_array, title='Max Amplitude Array', xlabel='X Index', ylabel='Y Index', zlabel='Amplitude')
    maxima_coords, maxima_values = find_prominent_maxima(max_amplitude_array, category, file_name)
    plot_peaks(max_amplitude_array, maxima_coords, maxima_values, title='Prominent Maxima in Smoothed Signal', xlabel='X Index', ylabel='Y Index')

    # initial guess for size of the particles according to labels
    n_size_clusters=len(size_labels)
    thresholds = calculate_thresholds(maxima_values, n_size_clusters)


    # Initialize list to store particle data
    maxima_labels = []


    # Loop through the prominent maxima
    for coord, value in zip(maxima_coords, maxima_values):
        print(f"Maxima at (X: {coord[0]}, Y: {coord[1]}) -> Amplitude: {value}")

        plot_heatmap(max_amplitude_array, coord)
        # Query for radius and material type
        # Replace these placeholder functions with the actual logic or arrays
        predictions = predict_amplitude_labels(thresholds, value, size_labels)


        while True:
            prompt = input(f"If radius is {predictions} press enter, else enter correct value? ")  # Function or array access to get radius at the coordinate

            if prompt == '0':
                break
            elif prompt == 'end':
                break
            elif prompt == "":
                radius = predictions
            elif prompt.isnumeric():
                radius = int(prompt)
            else:
                print("Invalid input. Please enter an integer. Enter 0 if you want to exclude this particle.")
                continue

            # Ask for material type
            while True:
                material_prompt = input(f"If the particle is {default_material} press enter, else enter correct label.")  # Function or array to get material type at the coordinate
                if material_prompt in ['pe', 'steel', 'glass', 'pmma', '']:
                    material_type = default_material if material_prompt == "" else material_prompt
                    break
                else:
                    print("Invalid input. Supported inputs are pe, steel, glass, pmma.")

        

            # Collect the signal at the maxima and its neighbors
            for i in range(-2,3):
                for j in range(-2,3):
                    if (coord[0]+i<=signal_array.shape[1]-1 and coord[1]+j<=signal_array.shape[2]-1):
                        signal_at_maxima=RF[:, coord[0]+i, coord[1]+j]
                        # Store the signal, radius, and material type in particle_info
                        particle_info = [signal_at_maxima, [coord[0]+i, coord[1]+j], radius, material_type]
                        maxima_labels.append(particle_info)
            break
        
    # Optional: Print out collected data for verification
    for particle in maxima_labels:
        print(f"\nMaxima detected with Radius: {particle[2]}, Material Type: {particle[3]}, Coordinates: {particle[1]}")
        print(f"  Signal: {particle[0]}")



    # Write to CSV file
    with open(csv_output, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write headers
        writer.writerow(["Signal", "Radius", "Material Type", "coordinates"])
        
        # Write data
        for particle_info in maxima_labels:
            # Convert the signal array to a string (if it's a numpy array or list)
            signal_as_string = ','.join(map(str, particle_info[0]))
            writer.writerow([signal_as_string, particle_info[2], particle_info[3], particle_info[1]])

    print(f"Data successfully saved to {csv_output}")

if __name__ == "__main__":
    main()
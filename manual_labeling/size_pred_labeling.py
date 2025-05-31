import numpy as np
from sklearn.cluster import KMeans
import os, sys



def train_and_predict_amplitude_classifier(amplitudes, n_classes, new_amplitudes):
    """
    Train a model to classify amplitude values using K-Means clustering, 
    and then predict new amplitude values.

    :param amplitudes: list or np.array of original amplitude values (training data)
    :param n_classes: Number of classes to classify the amplitudes into
    :param new_amplitudes: list or np.array of new amplitude values to classify
    :return: List of predicted class labels for the new_amplitudes
    """
    # Reshape the data as KMeans expects 2D data
    amplitudes = np.array(amplitudes).reshape(-1, 1)
    
    # Step 1: Train the model using KMeans
    kmeans = KMeans(n_clusters=n_classes, random_state=0)
    kmeans.fit(amplitudes)
    
    # Step 2: Predict the class labels for the new amplitudes
    new_amplitudes = np.array(new_amplitudes).reshape(-1, 1)
    predictions = kmeans.predict(new_amplitudes)
    
    return predictions

def calculate_thresholds(amplitudes, n_classes):
    """
    Calculate threshold values based on quantiles, ensuring each bin has approximately 
    the same number of amplitude values.
    
    :param amplitudes: list or np.array of original amplitude values (training data)
    :param n_classes: Number of classes (quantiles) to divide the data into
    :return: List of threshold values
    """
    # Calculate the quantile-based thresholds
    thresholds = np.quantile(amplitudes, np.linspace(0, 1, n_classes + 1))
    
    return thresholds

def predict_amplitude_labels(thresholds, amplitude, guide):
    """
    Assign class labels to new amplitude values based on predefined thresholds (quantiles).
    
    :param thresholds: List of calculated thresholds from calculate_thresholds
    :param new_amplitudes: List of new amplitude values to classify
    :return: List of predicted class labels for new_amplitudes
    """
    labels = []
    # Find the appropriate bin (class) based on thresholds
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= amplitude <= thresholds[i + 1]:
            labels.append(guide[i])  # Assign class label based on the bin
            break

    return labels


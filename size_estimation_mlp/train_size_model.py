'''
Navid.Zarrabi@torontomu.ca
May 2025
Goal: 
Train MLPs for size estimation from RF data
'''

import numpy as np
import pandas as pd
import yaml
import os
from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from size_model_arch import create_model
from util import clean_radius, get_bin_label

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_directories(config):
    """Create necessary output directories."""
    for dir_path in [config['output']['model_dir'], config['output']['plot_dir']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    # Load configuration
    config = load_config()
    
    # Setup output directories
    setup_directories(config)
    
    # Load and preprocess data
    print("Loading data...")
    root_dir = Path(__file__).parent
    data_path = root_dir.parent / config['data']['data_path']
    data = pd.read_csv(data_path)

    # Clean radius values
    data['Radius'] = data['Radius'].apply(clean_radius)

    # Convert signal strings to numpy arrays
    data['Signal'] = data['Signal'].apply(lambda x: np.fromstring(x, sep=','))

    # Pad signals to uniform length
    max_length = max(len(signal) for signal in data['Signal'])
    X = np.array([np.pad(signal, (0, max_length - len(signal))) for signal in data['Signal']])

    # Extract size bins from config
    bin_ranges = [(bin['min'], bin['max']) for bin in config['size_bins']]
    bin_labels = list(range(len(bin_ranges)))

    # Process each material separately
    print("\nProcessing materials...")
    for material in data['Material Type'].unique():
        print(f"\nProcessing material: {material}")
        
        # Filter data for current material
        material_mask = data['Material Type'] == material
        X_material = X[material_mask]
        y_material = data.loc[material_mask, 'Radius'].apply(lambda x: get_bin_label(x, bin_ranges))
        
        # Skip if not enough samples
        if len(X_material) < config['data']['min_samples']:
            print(f"Skipping {material} - not enough samples")
            continue
        
        # Remove samples that don't fall into any bin
        valid_indices = y_material.notna()
        X_material = X_material[valid_indices]
        y_material = y_material[valid_indices]
        
        if len(X_material) < config['data']['min_samples']:
            print(f"Skipping {material} - not enough valid samples after binning")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_material, y_material,
            test_size=config['training']['test_size'],
            random_state=config['training']['random_state']
        )
        
        # Create and train model
        model = create_model((X_train.shape[1],), config['model']['num_classes'])
        
        # Add early stopping callback
        early_stopping = EarlyStopping(
            monitor=config['training']['early_stopping']['monitor'],
            patience=config['training']['early_stopping']['patience'],
            restore_best_weights=config['training']['early_stopping']['restore_best_weights']
        )

        history = model.fit(
            X_train, y_train,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            validation_split=config['training']['validation_split'],
            verbose=1,
            callbacks=[early_stopping]
        )
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print results
        print(f"\nResults for {material}:")
        print(f"Accuracy: {np.mean(y_pred_classes == y_test):.4f}")
        
        # Print classification report with 4 decimal places
        print("\nClassification Report:")
        report = classification_report(
            y_test, y_pred_classes,
            labels=bin_labels,
            target_names=[f"Bin {i+1} ({min_size}-{max_size})" for i, (min_size, max_size) in enumerate(bin_ranges)],
            digits=4
        )
        print(report)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes, labels=bin_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f"Bin {i+1}" for i in range(len(bin_ranges))],
                    yticklabels=[f"Bin {i+1}" for i in range(len(bin_ranges))])
        plt.title(f'Confusion Matrix - {material}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # Save confusion matrix plot
        plot_path = Path(config['output']['plot_dir']) / f'confusion_matrix_{material}.png'
        plt.savefig(plot_path)
        plt.close()
        
        # Save the model
        model_path = Path(config['output']['model_dir']) / f'model_{material}.keras'
        model.save(model_path)
        print(f"Model saved as '{model_path}'")

if __name__ == '__main__':
    main()
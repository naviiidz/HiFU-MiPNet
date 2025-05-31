import os
import sys
# Add the parent directory to load prediction modules and parameters
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'material_classification_cnn')))
from config import get_training_config, preprocessing_parameters, predict_config, peak_extract_param


# Configuration
SAMPLING_RATE = 40e6  # Hz
LOWCUT_FREQ = 1e6    # Hz
HIGHCUT_FREQ = 20e6  # Hz
WINDOW_SIZE = 3      # Default window size
ALPHA = 0.8          # Default alpha
SIGMA = 3            # Default sigma
BOX_SIZE = 10        # Default box size

# Define available datasets by category
DATASETS = peak_extract_param()

def get_dataset_params(category, file_name=None):
    """Get the signal processing parameters for a specific dataset file."""
    try:
        if category not in DATASETS:
            print(f"Warning: Category '{category}' not found. Using default parameters.")
            return {
                "window_size": WINDOW_SIZE,
                "alpha": ALPHA,
                "sigma": SIGMA,
                "box_size": BOX_SIZE
            }

        if file_name and file_name in DATASETS[category]["files"]:
            return DATASETS[category]["files"][file_name]["params"]
        else:
            print(f"Warning: File '{file_name}' not found in category '{category}'. Using default parameters.")
            return {
                "window_size": WINDOW_SIZE,
                "alpha": ALPHA,
                "sigma": SIGMA,
                "box_size": BOX_SIZE
            }

    except Exception as e:
        print(f"Error getting dataset parameters: {str(e)}")
        return None




def get_dataset_path(category, file_name, root_dir):
    """Get the full path for a dataset file."""
    try:
        if category not in DATASETS:
            raise ValueError(f"Category '{category}' not found in available datasets")

        if file_name not in DATASETS[category]['files']:
            raise ValueError(f"File '{file_name}' not found in category '{category}'")

        return root_dir + DATASETS[category]['files'][file_name]['path']

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

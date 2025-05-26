import numpy as np  # For array operations
from scipy.ndimage import gaussian_filter  # For Gaussian smoothing
from config import peak_extract_param

#default values
DATASETS = peak_extract_param()
WINDOW_SIZE=5
ALPHA=1
SIGMA=1
BOX_SIZE=5

def find_prominent_maxima(max_amplitude_array, category=None, file_name=None):
    """Find local maxima in the amplitude array using file-specific parameters."""
    # Get parameters for the specific file
    params = get_dataset_params(category, file_name) if category else {
        "window_size": WINDOW_SIZE,
        "alpha": ALPHA,
        "sigma": SIGMA,
        "box_size": BOX_SIZE
    }

    window_size = params["window_size"]
    alpha = params["alpha"]
    sigma = params["sigma"]

    height, width = max_amplitude_array.shape

    # Stage 1: Enhanced noise reduction
    # Apply multiple Gaussian filters with different sigmas for multi-scale analysis
    smoothed_small = gaussian_filter(max_amplitude_array, sigma=sigma/2)
    smoothed_medium = gaussian_filter(max_amplitude_array, sigma=sigma)
    smoothed_large = gaussian_filter(max_amplitude_array, sigma=sigma*2)

    # Combine multi-scale smoothed images
    smoothed_array = (smoothed_small + smoothed_medium + smoothed_large) / 3

    # Stage 2: Adaptive thresholding
    # Calculate local statistics in sliding windows
    is_maxima = np.zeros_like(max_amplitude_array, dtype=bool)
    global_mean = np.mean(smoothed_array)
    global_std = np.std(smoothed_array)

    # Ensure window_size is odd
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    half_window = window_size // 2

    # Calculate dynamic threshold based on local and global statistics
    local_threshold = np.zeros_like(smoothed_array)
    for i in range(half_window, height - half_window):
        for j in range(half_window, width - half_window):
            window = smoothed_array[i-half_window:i+half_window+1,
                                  j-half_window:j+half_window+1]
            local_mean = np.mean(window)
            local_std = np.std(window)

            # Adaptive threshold combining local and global statistics
            threshold = (alpha * (local_mean + local_std) +
                       (1 - alpha) * (global_mean + global_std))
            local_threshold[i, j] = threshold

    # Stage 3: Non-maximum suppression with prominence check
    min_distance = window_size  # Minimum distance between peaks
    for i in range(half_window, height - half_window):
        for j in range(half_window, width - half_window):
            center_value = smoothed_array[i, j]
            window = smoothed_array[i-half_window:i+half_window+1,
                                  j-half_window:j+half_window+1]

            # Check if it's a local maximum
            if (center_value > local_threshold[i, j] and
                center_value == np.max(window)):

                # Calculate prominence (height above surrounding background)
                min_surrounding = np.min(window)
                prominence = center_value - min_surrounding

                # Only mark as maximum if prominence is significant
                if prominence > local_std:
                    is_maxima[i, j] = True

    # Stage 4: Filter peaks based on minimum distance
    maxima_coords = np.argwhere(is_maxima)
    maxima_values = smoothed_array[is_maxima]

    # Sort peaks by amplitude
    sort_idx = np.argsort(maxima_values)[::-1]
    maxima_coords = maxima_coords[sort_idx]
    maxima_values = maxima_values[sort_idx]

    # Filter peaks that are too close to stronger peaks
    filtered_coords = []
    filtered_values = []

    for i, (coord, value) in enumerate(zip(maxima_coords, maxima_values)):
        # Check if this peak is far enough from all stronger peaks
        if not filtered_coords or all(
            np.sqrt(np.sum((coord - prev_coord)**2)) >= min_distance
            for prev_coord in filtered_coords
        ):
            filtered_coords.append(coord)
            filtered_values.append(value)

    return np.array(filtered_coords), np.array(filtered_values)


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
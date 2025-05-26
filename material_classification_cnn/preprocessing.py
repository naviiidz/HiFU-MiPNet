import numpy as np
from scipy.signal import hilbert
from config import preprocessing_parameters

# Load preprocessing parameters when needed
params = preprocessing_parameters()

def process_signal(signal):
    signal = signal - np.mean(signal)
    window = np.hanning(len(signal))
    windowed_signal = signal * window

    analytic_signal = hilbert(windowed_signal)
    envelope = np.abs(analytic_signal)
    normalized_signal = (envelope - np.mean(envelope)) / np.std(envelope)

    fft_result = np.fft.fft(normalized_signal)
    magnitude_spectrum = np.abs(fft_result)
    phase_spectrum = np.angle(fft_result)

    if np.max(magnitude_spectrum) > 0:
        magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)

    complex_spectrum = magnitude_spectrum * np.exp(1j * phase_spectrum)
    return complex_spectrum


def prepare_balanced_dataset(signals, labels):
    """Create a balanced dataset with equal samples per class."""
    try:
        unique_labels = np.unique(labels)
        balanced_signals = []
        balanced_labels = []

        print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        print(f"Initial dataset size: {len(signals)} signals")

        # First, find the maximum length among all signals
        max_len = max(len(signal) for signal in signals)
        print(f"Maximum signal length: {max_len}")

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            print(f"\nProcessing label: {label}")
            print(f"Found {len(indices)} samples")

            if len(indices) >= params["samples_per_class"]:
                selected_indices = np.random.choice(indices, params["samples_per_class"], replace=False)
                print(f"Selected {params["samples_per_class"]} samples")
            else:
                print(f"Warning: Only {len(indices)} samples available for {label}")
                selected_indices = indices

            # Process and pad each selected signal
            for idx in selected_indices:
                if not isinstance(signals[idx], np.ndarray):
                    print(f"Warning: Invalid signal at index {idx}")
                    continue

                # Pad signal if necessary
                signal = signals[idx]
                if len(signal) < max_len:
                    padded_signal = np.pad(signal,
                                         (0, max_len - len(signal)),
                                         mode='constant',
                                         constant_values=0)
                else:
                    padded_signal = signal

                balanced_signals.append(padded_signal)
                balanced_labels.append(label)

        # Convert to numpy arrays
        balanced_signals = np.stack(balanced_signals)
        balanced_labels = np.array(balanced_labels)

        print(f"\nFinal balanced dataset size: {len(balanced_signals)} signals")
        print(f"Signal shape: {balanced_signals.shape}")

        return balanced_signals, balanced_labels

    except Exception as e:
        print(f"Error in prepare_balanced_dataset: {str(e)}")
        print("Detailed signal information:")
        print(f"Number of signals: {len(signals)}")
        print("Signal lengths:", [len(s) for s in signals[:5]], "... (first 5 shown)")
        print(f"Labels shape: {labels.shape}")
        raise
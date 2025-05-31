from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set global font size
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

def show_3d_max(max_amplitude_array, title='Max Amplitude Array', xlabel='X Index', ylabel='Y Index', zlabel='Amplitude'):
    # # 3D Visualization
    X = np.arange(max_amplitude_array.shape[0])
    Y = np.arange(max_amplitude_array.shape[1])
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, max_amplitude_array.T, cmap='viridis')

    ax.set_title('3D Visualization of Aggregated Signals')
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    ax.set_zlabel('Signal Amplitude')

    plt.tight_layout()
    plt.show()

    # 3D Visualization of Shifted Signals
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, max_amplitude_array.T, cmap='viridis')

    ax.set_title('3D Visualization of Shifted Signals (Peaks Shifted Above)')
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    ax.set_zlabel('Signal Amplitude')

    plt.tight_layout()
    plt.show()

def plot_peaks(max_amplitude_array, maxima_coords, maxima_values, title='Prominent Maxima', xlabel='X Index', ylabel='Y Index'):
    # # 3D Visualization
    X = np.arange(max_amplitude_array.shape[0])
    Y = np.arange(max_amplitude_array.shape[1])
    X, Y = np.meshgrid(X, Y)

    # 3D Visualization of Smoothed Signals with Prominent Maxima
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, max_amplitude_array.T, cmap='viridis')

    # Plot maxima points on the surface
    ax.scatter(maxima_coords[:, 0], maxima_coords[:, 1], max_amplitude_array[maxima_coords[:, 0], maxima_coords[:, 1]], color='red', s=50, label='Prominent Maxima')

    ax.set_title(f'3D Visualization of Smoothed Signals with Prominent Maxima')
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    ax.set_zlabel('Signal Amplitude')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_heatmap(max_amplitude_array, coord):
    
    # Plot the heatmap and mark the maxima
    plt.figure(figsize=(6, 6))
    sns.heatmap(max_amplitude_array, cmap='viridis', cbar=True)
    
    # Mark the maxima with a red star
    plt.plot(coord[1], coord[0], 'r*', markersize=15)  # coord[1] is Y, coord[0] is X
    
    plt.title(f"Maxima at (X: {coord[0]}, Y: {coord[1]})")
    plt.xlabel("Y Index")
    plt.ylabel("X Index")
    
    plt.show()  
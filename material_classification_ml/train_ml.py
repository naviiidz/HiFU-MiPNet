"""
Author: Navid Zarrabi (Navid.Zarrabi@torontomu.ca)
Date: May 26, 2025
Purpose: Train traditional ML models for material identification of microspheres.

Key Notes:
1. Modular design with separate functions for data loading, preprocessing, and model training
2. Configuration-driven approach using YAML
3. Comprehensive model evaluation and visualization
4. Support for multiple random seeds for robust evaluation
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from feature_extraction import fe_time, fe_freq, cal_corr, show_corr
import pickle

@dataclass
class ModelConfig:
    """Configuration for model training."""
    num_seeds: int
    test_size: float
    random_state: int
    feature_correlation_threshold: float
    data_dir: Path
    output_dir: Path

@dataclass
class ModelResult:
    """Container for model evaluation results."""
    accuracy: float
    std_accuracy: float
    precision: float
    std_precision: float
    recall: float
    std_recall: float
    f1: float
    std_f1: float
    class_accuracies: np.ndarray
    std_class_accuracies: np.ndarray
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float] = None

def load_config() -> ModelConfig:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert output_dir to Path object
    config['training']['output_dir'] = Path(__file__).parent / config['training']['output_dir']
    return ModelConfig(**config['training'])

def setup_directories(config: ModelConfig) -> None:
    """Create necessary output directories."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / 'plots').mkdir(exist_ok=True)
    (config.output_dir / 'models').mkdir(exist_ok=True)

def load_and_preprocess_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess the dataset."""
    print("\nLoading data...")
    data = pd.read_csv(data_path)
    
    # Extract signals and material types
    signals = data['Signal'].values
    material_types = data['Material Type'].values
    
    # Convert signals from strings to numpy arrays
    signals_array = [np.fromstring(signal, sep=',') for signal in signals]
    
    # Apply FFT to all signals
    signals_freq_array = [signal_to_frequency(signal) for signal in signals_array]
    
    # Extract features
    features_list = []
    for i in range(len(signals_array)):
        time_features = fe_time(signals_array[i])
        freq_features = fe_freq(signals_freq_array[i])
        all_features = list(time_features.values()) + list(freq_features.values())
        features_list.append(all_features)
    
    # Get feature names
    time_feature_names = [f"time_{k}" for k in fe_time(signals_array[0]).keys()]
    freq_feature_names = [f"freq_{k}" for k in fe_freq(signals_freq_array[0]).keys()]
    feature_names = time_feature_names + freq_feature_names
    
    return np.array(features_list), material_types, feature_names

def signal_to_frequency(signal: np.ndarray) -> np.ndarray:
    """Convert signal to frequency domain."""
    fft_result = np.fft.fft(signal)
    return np.abs(fft_result)

def get_models(seed: int) -> Dict[str, Any]:
    """Initialize all models to be evaluated."""
    return {
        'Random Forest': RandomForestClassifier(random_state=seed),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(random_state=seed, max_iter=500),
        'Gradient Boosting': GradientBoostingClassifier(random_state=seed),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 100), random_state=seed, max_iter=500),
        'Decision Tree': DecisionTreeClassifier(random_state=seed),
        'SVM (RBF Kernel)': SVC(kernel='rbf', random_state=seed),
        'Voting Classifier': VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(random_state=seed)),
            ('lr', LogisticRegression(random_state=seed, max_iter=500)),
            ('gbm', GradientBoostingClassifier(random_state=seed)),
        ], voting='soft')
    }

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                  label_encoder: LabelEncoder) -> ModelResult:
    """Evaluate a trained model and return results."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(label_encoder.classes_, model.feature_importances_))
    
    return ModelResult(
        accuracy=accuracy,
        std_accuracy=0.0,  # Will be updated later
        precision=report['weighted avg']['precision'],
        std_precision=0.0,
        recall=report['weighted avg']['recall'],
        std_recall=0.0,
        f1=report['weighted avg']['f1-score'],
        std_f1=0.0,
        class_accuracies=class_accuracies,
        std_class_accuracies=np.zeros_like(class_accuracies),
        confusion_matrix=cm,
        feature_importance=feature_importance
    )

def plot_results(results: Dict[str, ModelResult], output_dir: Path) -> None:
    """Plot and save evaluation results."""
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    accuracies = [r.accuracy for r in results.values()]
    std_accuracies = [r.std_accuracy for r in results.values()]
    
    plt.bar(models, accuracies, yerr=std_accuracies)
    plt.xticks(rotation=45)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'accuracy_comparison.png')
    plt.close()
    
    # Plot confusion matrices
    for model_name, result in results.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(result.confusion_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Average Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(output_dir / 'plots' / f'confusion_matrix_{model_name}.png')
        plt.close()

def run_experiment(seed: int, features: np.ndarray, labels: np.ndarray,
                  feature_names: List[str], config: ModelConfig) -> Tuple[Dict[str, ModelResult], LabelEncoder]:
    """Run a single experiment with the given random seed."""
    print(f"\n--- Running experiment with random seed: {seed} ---")
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=config.test_size, random_state=seed
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Initialize models
    models = get_models(seed)
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train_encoded)
        results[name] = evaluate_model(model, X_test_scaled, y_test_encoded, label_encoder)
        
        # Save model
        model_path = config.output_dir / 'models' / f'{name}_seed_{seed}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    return results, label_encoder

def main():
    """Main function to run the training pipeline."""
    # Load configuration
    config = load_config()
    setup_directories(config)
    
    # Load and preprocess data
    data_path = Path(__file__).parent.parent / config.data_dir
    features, labels, feature_names = load_and_preprocess_data(str(data_path))
    
    # Generate random seeds
    np.random.seed(config.random_state)
    random_seeds = np.random.randint(0, 10000, size=config.num_seeds)
    
    # Run experiments
    all_results = {}
    label_encoder = None
    for seed in random_seeds:
        seed_results, label_encoder = run_experiment(seed, features, labels, feature_names, config)
        
        # Aggregate results
        for model_name, result in seed_results.items():
            if model_name not in all_results:
                all_results[model_name] = []
            all_results[model_name].append(result)
    
    # Calculate average results
    final_results = {}
    for model_name, results in all_results.items():
        final_results[model_name] = ModelResult(
            accuracy=np.mean([r.accuracy for r in results]),
            std_accuracy=np.std([r.accuracy for r in results]),
            precision=np.mean([r.precision for r in results]),
            std_precision=np.std([r.precision for r in results]),
            recall=np.mean([r.recall for r in results]),
            std_recall=np.std([r.recall for r in results]),
            f1=np.mean([r.f1 for r in results]),
            std_f1=np.std([r.f1 for r in results]),
            class_accuracies=np.mean([r.class_accuracies for r in results], axis=0),
            std_class_accuracies=np.std([r.class_accuracies for r in results], axis=0),
            confusion_matrix=np.mean([r.confusion_matrix for r in results], axis=0),
            feature_importance=results[0].feature_importance
        )
    
    # Plot and save results
    plot_results(final_results, config.output_dir)
    
    # Print final results
    print("\n" + "="*50)
    print(f"AVERAGE RESULTS ACROSS {config.num_seeds} RANDOM SEEDS")
    print("="*50)
    
    for model_name, result in final_results.items():
        print(f"\n{model_name}:")
        print(f"Average Accuracy: {result.accuracy:.4f} ± {result.std_accuracy:.4f}")
        print(f"Average Precision: {result.precision:.4f} ± {result.std_precision:.4f}")
        print(f"Average Recall: {result.recall:.4f} ± {result.std_recall:.4f}")
        print(f"Average F1 Score: {result.f1:.4f} ± {result.std_f1:.4f}")
        
        print("Class-wise Accuracy:")
        for idx, (avg_acc, std_acc) in enumerate(zip(result.class_accuracies, result.std_class_accuracies)):
            class_name = label_encoder.inverse_transform([idx])[0]
            print(f"  Class {class_name}: {avg_acc:.4f} ± {std_acc:.4f}")

if __name__ == "__main__":
    main()
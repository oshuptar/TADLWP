import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Dict, Any


def generate_simple_line_seperated_dataset(n_samples: int = 200, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a simple binary classification dataset (line).
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        X: Input features (N, 2)
        y: True labels (N,)
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, 2)
    # Simple line: y = x (points above the line y=x are class 1)
    y = (X[:, 1] > X[:, 0]).float()
    return X, y


def generate_spiral_dataset(n_samples: int = 1000, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a complex spiral dataset for binary classification.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        X: Input features (N, 2)
        y: True labels (N,)
    """
    torch.manual_seed(seed)
    t = torch.linspace(0, 4 * np.pi, n_samples)
    r = t / (4 * np.pi)
    x1 = r * torch.cos(t) + 0.1 * torch.randn(n_samples)
    x2 = r * torch.sin(t) + 0.1 * torch.randn(n_samples)
    X = torch.stack([x1, x2], dim=1)
    y = (t > 2 * np.pi).float()
    return X, y


def generate_pixels_dataset(image_path: str='image.png', size_reduction: int=30) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load an image from disk.
    Generate a dataset of pixel positions mapped to image intensity.
    
    Args:
        image_path: Path to the image
        size_reduction: Takes every n-th pixel

    Returns:
        X: Input features ((H//size_reduction)*(W//size_reduction), 2)
        y: True labels (H//size_reduction*W//size_reduction, )
    """
    img = Image.open(image_path)
    grayscale_img = img.convert('L')
    img_array = np.array(grayscale_img)
    img_array = np.rot90(np.rot90(np.rot90(img_array)))
    X = []
    y = []
    for i in range(0, img_array.shape[0], 30):
        for j in range(0, img_array.shape[1], 30):
            X.append([i / img_array.shape[0] * 2.0 - 1.0, j / img_array.shape[1] * 2.0 - 1.0])
            y.append(img_array[i][j] / 255.0 * 2.0 - 1.0)
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    return X, y


def visualize_dataset(X: torch.Tensor, y: torch.Tensor, title: str = "Dataset") -> None:
    """
    Visualize the dataset with true labels.
    
    Args:
        X: Input features (N, 2)
        y: True labels (N,)
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7, s=20)
    plt.title(title)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_predictions(X: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, title: str = "Model Predictions") -> None:
    """
    Visualize model predictions compared to true labels.
    
    Args:
        X: Input features (N, 2)
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # True labels
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='coolwarm', alpha=0.7, s=20)
    ax1.set_title('True Labels')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Class')
    
    # Predicted labels
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', alpha=0.7, s=20)
    ax2.set_title('Model Predictions')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Class')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_training_progress(losses: List[float], title: str = "Training Progress") -> None:
    """
    Plot the training loss over epochs.
    
    Args:
        losses: List of loss values
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def evaluate_model(model: Any, X: torch.Tensor, y: torch.Tensor):
    """
    Evaluate model accuracy on the dataset.
    
    Args:
        model: Trained PyTorch model
        X: Input features
        y: True labels
        
    Returns:
        accuracy: Model accuracy
        predictions: Model predictions
    """
    with torch.no_grad():
        predictions = (model(X).squeeze() > 0.5).float()
        accuracy = (predictions == y).float().mean()
    return accuracy, predictions

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import List
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from helpers.training_utils import load_yeast_data, divide_data_to_train_val_test, evaluate_model_on_test, evaluate, plot_training_history
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from part_2 import (
    run_experiments
)

"""
TODO: Implement weight initialization methods and hyperparameter tuning.

In this part, you will implement:
1. Weight initialization - implement different initialization strategies (Xavier uniform, Kaiming uniform, uniform, normal)
2. Hyperparameter lists - create lists of learning rates, batch sizes, and momentum values to test
3. Experiment configuration - use the best methods from Part 2 and test different weight initializations

Key concepts:
- Weight initialization - proper initialization is crucial for training deep networks
  - Xavier uniform: Good for layers with tanh/sigmoid activations
  - Kaiming uniform: Good for layers with ReLU activations
  - Uniform/Normal: Simple random initialization
- Hyperparameter tuning - systematically test different combinations to find optimal settings
- Use the best balancing method from Part 2 and combine it with different weight initializations

After implementation, run experiments to find the best combination of weight initialization and hyperparameters.
"""

def init_model_xavier(model: nn.Module) -> None:
    """
    TODO: Implement Xavier uniform initialization for Linear layers.
    Hint: nn.init.xavier_uniform_(module.weight), nn.init.constant_(module.bias, 0.0)
    """
    raise NotImplementedError("TODO: Implement init_model_xavier")


def init_model_kaiming(model: nn.Module) -> None:
    """
    TODO: Implement Kaiming uniform initialization for Linear layers (ReLU).
    Hint: nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
    """
    raise NotImplementedError("TODO: Implement init_model_kaiming")


def init_model_uniform(model: nn.Module) -> None:
    """
    TODO: Implement uniform initialization for Linear layers (e.g. -0.1, 0.1).
    Hint: nn.init.uniform_(module.weight, -0.1, 0.1)
    """
    raise NotImplementedError("TODO: Implement init_model_uniform")


def init_model_normal(model: nn.Module) -> None:
    """
    TODO: Implement normal initialization for Linear layers (mean=0, std=0.1).
    Hint: nn.init.normal_(module.weight, 0.0, 0.1)
    """
    raise NotImplementedError("TODO: Implement init_model_normal")


def create_learning_rate_list() -> List[float]:
    """
    TODO: Implement learning rate list
    """
    raise NotImplementedError("TODO: Implement learning rate list")

def create_batch_size_list() -> List[int]:
    """
    TODO: Implement batch size list
    """
    raise NotImplementedError("TODO: Implement batch size list")

def create_momentum_list() -> List[float]:
    """
    TODO: Implement momentum list
    """
    raise NotImplementedError("TODO: Implement momentum list")


def main():
    run_experiments(
        balance_method_list=['weighted_sampler'],
        weight_initialization_list=[
            init_model_xavier,
            init_model_kaiming,
            init_model_uniform,
            init_model_normal,
        ],
        learning_rate_list=create_learning_rate_list(),
        batch_size_list=create_batch_size_list(),
        momentum_list=create_momentum_list(),
        seed=42,
        verbose=False
    )



if __name__ == "__main__":
    main()

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
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight);
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0); 


def init_model_kaiming(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu');
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0);
    


def init_model_uniform(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -0.1, 0.1);
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0);


def init_model_normal(model: nn.Module) -> None:
    """
    TODO: Implement normal initialization for Linear layers (mean=0, std=0.1).
    Hint: nn.init.normal_(module.weight, 0.0, 0.1)
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.1);
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0);


def create_learning_rate_list() -> List[float]:
    return [multiplier*10**(exp) 
            for exp in range (-3,0) 
            for multiplier in [1,2,3]];

def create_batch_size_list() -> List[int]:
    return [2 ** exp for exp in range(3, 8)];
    

def create_momentum_list() -> List[float]:
    return [0.0, 0.9, 0.95, 0.99];


def main():
    run_experiments(
        balance_method_list=[None, 'oversample', 'class_weight', 'weighted_sampler'],
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

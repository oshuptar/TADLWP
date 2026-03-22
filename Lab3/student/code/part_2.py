import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from itertools import product
from imblearn.over_sampling import RandomOverSampler
from helpers.training_utils import set_random_seed, print_experiment_results
from part_1 import run_experiment, create_basic_criterion, create_data_loaders

"""
TODO: Implement methods for handling imbalanced datasets.

In this part, you will implement three different techniques to handle class imbalance:
1. Oversampling - duplicate minority class samples to balance the dataset using RandomOverSampler
2. Class-weighted loss - modify the loss function to penalize minority class errors more heavily
3. Weighted sampling - use WeightedRandomSampler to balance batches during training

Key concepts:
- RandomOverSampler from imblearn - creates synthetic samples by duplicating minority class examples
- Class weights calculation - compute balanced weights using formula: weight = total_samples / (n_classes * samples_per_class)
- WeightedRandomSampler - samples data with probabilities proportional to class weights
- Grid search - test different combinations of methods to find the best approach

After implementation, run experiments to compare which method works best for the imbalanced MNIST dataset.
"""

def _calculate_class_weights(y_train: np.ndarray) -> Tuple[torch.Tensor, Dict[int, float]]:
    """Compute balanced class weights: total / (n_classes * count_per_class).
    Returns tuple: first part (tensor) for nn.CrossEntropyLoss(weight=...) so minority
    class errors contribute more to the loss; second part (dict class_index -> weight)
    for per-sample weights in WeightedRandomSampler."""
    counts = pd.Series(y_train).value_counts().sort_index()
    weights = len(y_train) / (len(counts) * counts.values)
    return torch.from_numpy(weights.astype('float32')), dict(zip(counts.index, weights))


########################################################
# 3 Methods for handling unbalanced data:
########################################################

# Method 1: Oversampling - duplicate minority class samples to match majority
def apply_oversampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    randomOverSampler = RandomOverSampler(sampling_strategy="auto", random_state=42)
    resampled: Tuple[np.ndarray, np.ndarray] = randomOverSampler.fit_resample(X_train, y_train)
    X_resampled, y_resampled = resampled
    return X_resampled, y_resampled

# Method 2: Class-weighted loss - penalize minority class errors more
def create_class_weighted_criterion(y_train: np.ndarray, device: Optional[torch.device] = None) -> nn.CrossEntropyLoss:
    weights, _ =_calculate_class_weights(y_train=y_train)
    weights.to(device=device);
    return nn.CrossEntropyLoss(weight=weights)


# Method 3: Weighted sampling - balance each training batch
def create_weighted_sampler_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long());
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long());
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long());

    _, class_to_weight = _calculate_class_weights(y_train=y_train);
    weights = [class_to_weight[c] for c in y_train]
    weightedRandomSampler = WeightedRandomSampler(weights=weights, num_samples=len(weights));

    train_loader = DataLoader(dataset=train_dataset, sampler=weightedRandomSampler, shuffle=False, batch_size=batch_size);
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size);
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size);
    return train_loader, val_loader, test_loader


########################################################
# Experiment runners:
########################################################

def starting_experiment(
    balance_method: Optional[str],
    weight_init: Optional[Callable[[nn.Module], None]] = None,
    lr: float = 0.02,
    batch_size: int = 64,
    momentum: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single experiment. balance_method: None, 'oversample', 'class_weight', or 'weighted_sampler'."""
    set_random_seed(seed)
    return run_experiment(
        create_data_loaders=create_weighted_sampler_data_loaders if balance_method == 'weighted_sampler' else create_data_loaders,
        criterion=create_class_weighted_criterion if balance_method == 'class_weight' else create_basic_criterion,
        data_processor=(lambda X, y: apply_oversampling(X, y, seed, verbose)) if balance_method == 'oversample' else None,
        weight_initialization=weight_init,
        learning_rate=lr,
        batch_size=batch_size,
        momentum=momentum,
        verbose=verbose
    )


def run_experiments(
    balance_method_list: List[Optional[str]],
    weight_initialization_list: List[Optional[Callable[[nn.Module], None]]],
    seed: int = 42,
    learning_rate_list: Optional[List[float]] = None,
    batch_size_list: Optional[List[int]] = None,
    momentum_list: Optional[List[float]] = None,
    verbose: bool = True,
) -> None:
    """Run grid search; each run uses exactly one balancing method (or none)."""
    lr_list = learning_rate_list or [0.02]
    bs_list = batch_size_list or [64]
    mom_list = momentum_list or [0.0]
    results = []
    param_grid = product(balance_method_list, weight_initialization_list, lr_list, bs_list, mom_list)
    for balance_method, weight_init, lr, bs, momentum in param_grid:
        print(f"\nTesting: balance_method={balance_method}, lr={lr}, batch_size={bs}, momentum={momentum}")
        result = starting_experiment(balance_method, weight_init, lr, bs, momentum, seed, verbose)
        results.append({
            'balance_method': balance_method, 'weight_initialization': weight_init,
            'learning_rate': lr, 'batch_size': bs, 'momentum': momentum,
            'f1_macro': result['metrics']['f1_macro'], 'test_acc': result['metrics']['accuracy']
        })
        print(f"F1 Macro: {result['metrics']['f1_macro']:.4f}")
    print_experiment_results(results, balance_method_list, weight_initialization_list,
                             lr_list, bs_list, mom_list, top_n=5)


def main():
    run_experiments(
        balance_method_list=[None, 'oversample', 'class_weight', 'weighted_sampler'],
        weight_initialization_list=[None],
        seed=42,
        verbose=False
    )


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from helpers.training_utils import (
    load_unbalanced_mnist,
    divide_data_to_train_val_test,
    evaluate,
    # Logging helpers (hide experiment infrastructure complexity)
    setup_epoch_logging,
    log_batch_step,
    finalize_epoch_logging,
    log_epoch_metrics,
    create_experiment_with_config,
    finalize_training,
)


"""
TODO: Implement the main training functions for a neural network model.

In this part, you will implement:
1. Model architecture - create a feedforward neural network using PyTorch's Sequential and Linear layers
2. Data loaders - create PyTorch DataLoaders for training, validation, and test sets
3. Training loop - implement the complete training loop with forward pass, loss calculation, backward pass, and optimizer step
4. Validation evaluation - implement validation evaluation using model.eval() and torch.no_grad()

Key concepts:
- Use nn.Sequential to build multi-layer networks
- Use TensorDataset and DataLoader for efficient data handling
- Implement training loop: model.train(), optimizer.zero_grad(), forward pass, loss.backward(), optimizer.step()
- Implement validation: model.eval(), torch.no_grad(), calculate accuracy
- Track training history (loss and accuracy) for both training and validation sets
"""




def create_model(input_size: int, num_classes: int) -> nn.Module:
    """
    TODO: Implement the model with Dense layers 
    Hint: You can use Sequential, nn.Linear and nn.ReLU, around 3-4 layers should be enough
    """
    raise NotImplementedError("TODO: Implement the model with Dense layers using Sequential")
    return model


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """
    TODO: Implement the data loaders
    Hint: You can use TensorDataset and DataLoader
    """
    raise NotImplementedError("TODO: Implement the data loaders")
    return train_loader, val_loader, test_loader



def train_model_with_history(
    experiment: Any,
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    criterion: nn.Module,
    learning_rate: float = 0.02,
    momentum: float = 0.0,
    epochs: int = 100,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        verbose_epoch = epoch % 10 == 0 or epoch == epochs
        running_loss = 0.0
        correct = 0
        total = 0
        if verbose_epoch:
            hook_handles, parameters_from_epoch_start, old_parameters = setup_epoch_logging(model, experiment)
        for inputs, targets in train_loader:
            """
            TODO: Implement the training iteration
            Hint: You can use optimizer.zero_grad(), optimizer.step(), loss.backward()
            """
            raise NotImplementedError("TODO: Implement the training iteration")
            if verbose_epoch:
                old_parameters = log_batch_step(old_parameters, model, experiment)
        # TODO: calculate train loss and train accuracy from running_loss, correct, total
        train_loss = 0.0  # replace: running_loss / len(train_loader.dataset)
        train_acc = 0.0  # replace: correct / total
        if verbose_epoch:
            finalize_epoch_logging(hook_handles, parameters_from_epoch_start, old_parameters, experiment)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if verbose_epoch:
            print(f"Epoch {epoch}/{epochs} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            log_epoch_metrics(experiment, train_loss, val_loss, val_acc, model)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    return model, history


def create_basic_criterion(_, device=None):
    # This function will simplify the code in part 2
    return nn.CrossEntropyLoss()

def empty_data_processor(X_train, y_train):
    # This function will simplify the code in part 2
    return X_train, y_train


def run_experiment(create_data_loaders, 
                   criterion, 
                   data_processor=None, 
                   weight_initialization=None,
                   learning_rate=0.02,
                   batch_size=32,
                   momentum=0.0,
                   verbose=True):
    # Load and prepare data
    df = load_unbalanced_mnist()
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])
    class_names = label_encoder.classes_ 
    X_train, X_val, X_test, y_train, y_val, y_test = divide_data_to_train_val_test(df, test_size=0.2, val_size=0.1)
    X_train, y_train = (data_processor or empty_data_processor)(X_train, y_train)

    # Detect device first so criterion weights can be on the same device (e.g. for class-weighted loss)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = criterion(y_train, device=device)

    # Create data loaders and model
    train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size)
    model = create_model(X_train.shape[1], len(class_names))
    
    if weight_initialization is not None:
        weight_initialization(model)

    # Setup experiment
    experiment = create_experiment_with_config(learning_rate, momentum, batch_size, weight_initialization)

    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    
    # Move model to device
    model = model.to(device)
    
    # Train the model
    model, history = train_model_with_history(experiment, model, train_loader, val_loader, criterion, learning_rate=learning_rate, momentum=momentum, epochs=50, device=device)
    
    # Evaluate and visualize results
    result = finalize_training(model, history, test_loader, criterion, class_names, "Part 1 training history", verbose)
    
    return result


def main():
    run_experiment(
        create_data_loaders,
        criterion=create_basic_criterion,
        data_processor=empty_data_processor,
        verbose=True
    )


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import requests
import io
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import torch
import torch.nn as nn

from .experiment_logger import Experiment


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yeast_data():
    """Load Yeast data from UCI and return a DataFrame."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    # UCI dataset is whitespace-separated; column names per description
    col_names = ['SequenceName', 'mcg','gvh','alm','mit','erl','pox','vac','nuc','class']
    # Fetch URL content and load into DataFrame
    content = requests.get(url).content
    df = pd.read_csv(io.BytesIO(content), sep='\s+', names=col_names)
    df.drop('SequenceName', axis=1, inplace=True)

    # Drop classes with fewer than 100 samples
    class_counts = df['class'].value_counts()
    classes_to_keep = class_counts[class_counts >= 100].index
    df = df[df['class'].isin(classes_to_keep)].copy()
    
    return df


def load_unbalanced_mnist(imbalance_factor=0.96, max_samples_per_class=1000, random_state=42, noise_factor=0.3, verbose=False):
    """
    Load MNIST dataset with artificial class imbalance.
    
    Args:
        imbalance_factor: Float between 0 and 1. Controls the imbalance ratio.
            - imbalance_factor=0: All classes have max_samples_per_class (balanced)
            - imbalance_factor=1: Class 0 has 3 samples, class 9 has max_samples_per_class
            - Exponential interpolation between for intermediate classes
            - Example: imbalance_factor=1, class 0=3, class 5≈95, class 9=1000
        max_samples_per_class: Maximum number of samples per class (for class 9 when imbalance_factor=1)
        random_state: Random seed for reproducibility
        noise_factor: Float between 0 and 1. Adds randomness to sample counts (default 0.1 = ±10%)
    
    Returns:
        DataFrame with flattened pixel features and 'class' column
    """
    from sklearn.datasets import fetch_openml
    
    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    np.random.seed(random_state)
    
    num_classes = 10
    
    # Create imbalanced dataset
    X_imbalanced = []
    y_imbalanced = []
    
    for cls in range(num_classes):
        # Get indices for this class
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        
        # Calculate base samples using exponential interpolation based on class index
        # When imbalance_factor=1: class 0 -> 3 samples, class 9 -> max_samples_per_class
        # When imbalance_factor=0: all classes -> max_samples_per_class
        min_samples = 3  # Minimum samples for class 0 at full imbalance
        
        # Exponential interpolation: min_samples * (max/min)^(cls/(num_classes-1))
        ratio = max_samples_per_class / min_samples
        exponent = cls / (num_classes - 1)
        samples_at_full_imbalance = min_samples * (ratio ** exponent)
        
        # Blend between balanced and imbalanced based on imbalance_factor
        base_samples = max_samples_per_class * (1 - imbalance_factor) + imbalance_factor * samples_at_full_imbalance
        
        # Add random noise to make it more varied
        noise = np.random.uniform(-noise_factor, noise_factor) * base_samples
        n_samples = int(max(min_samples, base_samples + noise))  # Ensure at least min_samples
        n_samples = min(n_samples, len(cls_indices))  # Don't exceed available samples
        
        # Select samples
        selected_indices = cls_indices[:n_samples]
        X_imbalanced.append(X[selected_indices])
        y_imbalanced.append(y[selected_indices])
    
    X_imbalanced = np.vstack(X_imbalanced)
    y_imbalanced = np.concatenate(y_imbalanced)
    
    # Create DataFrame
    feature_names = [f'pixel_{i}' for i in range(X_imbalanced.shape[1])]
    df = pd.DataFrame(X_imbalanced, columns=feature_names)
    df['class'] = y_imbalanced
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Print class distribution
    if verbose:
        print("Unbalanced MNIST class distribution:")
        class_counts = df['class'].value_counts().sort_index()
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count} samples")
        
    return df

def divide_data_to_x_and_y(df):
    x = df.drop('class', axis=1).values
    y = df['class'].values
    return x, y


def divide_data_to_train_val_test(df, test_size=0.3, val_size=0.1):
    x, y = divide_data_to_x_and_y(df)
    train_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        x, y, test_size=(1.0 - train_size), stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(test_size/(test_size + val_size)), stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

########################################################
## Helper functions for evaluation:

def evaluate(model, data_loader, criterion, device='cpu'):
    """Evaluate model on the given DataLoader. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    acc = correct / total
    return avg_loss, acc

def compute_metrics_dict(y_true, y_pred, y_prob, class_names):
    """Compute metrics and return them as a dictionary."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Compute mean ROC AUC (macro)
    n_classes = len(class_names)
    roc_auc_scores = []
    for i in range(n_classes):
        y_bin = [1 if label == i else 0 for label in y_true]
        if sum(y_bin) > 0:  # only if class is present in test set
            roc_auc_scores.append(roc_auc_score(y_bin, [p[i] for p in y_prob]))
    roc_auc_macro = np.mean(roc_auc_scores) if roc_auc_scores else 0.0
    
    return {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'precision_weighted': prec_weighted,
        'recall_macro': rec_macro,
        'recall_weighted': rec_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc_macro': roc_auc_macro
    }

def compute_metrics(y_true, y_pred, y_prob, class_names):
    """Compute and print final metrics: accuracy, precision, recall, F1 (macro/weighted), confusion matrix, ROC AUC."""
    metrics = compute_metrics_dict(y_true, y_pred, y_prob, class_names)
    
    # print("\nAccuracy: {:.4f}".format(metrics['accuracy']))
    # print(f"Precision: macro={metrics['precision_macro']:.4f}, weighted={metrics['precision_weighted']:.4f}")
    # print(f"Recall:    macro={metrics['recall_macro']:.4f}, weighted={metrics['recall_weighted']:.4f}")
    # print(f"F1-score:  macro={metrics['f1_macro']:.4f}, weighted={metrics['f1_weighted']:.4f}")
    # print(f"ROC AUC (macro): {metrics['roc_auc_macro']:.4f}")
    
    # # Full classification report (metrics per class)
    # print("\n**Classification report (per class)**:\n", 
    #       classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion matrix – visualization as figure
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix as a figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Sample count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    n_classes = len(class_names)
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        # Binary label: whether sample belongs to class i (True/False)
        y_bin = [1 if label == i else 0 for label in y_true]
        fpr[i], tpr[i], _ = roc_curve(y_bin, [p[i] for p in y_prob])
        roc_auc[i] = roc_auc_score(y_bin, [p[i] for p in y_prob])
    # ROC plot
    plt.figure(figsize=(8,6))
    for i, class_name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_name} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0,1],[0,1],'--', color='gray')  # diagonal for random classifier
    plt.title("ROC Curve - One-vs-Rest")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

def evaluate_model_on_test(model, test_loader, criterion, class_names, method_name, verbose=True):
    """
    Common function to evaluate model on the test set.
    
    Args:
        model: trained model
        test_loader: DataLoader for test set
        criterion: loss function
        class_names: class names
        method_name: method name (for result)
        verbose: whether to print details
    
    Returns:
        dict with keys 'method' and 'metrics'
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            probs = nn.functional.softmax(outputs, dim=1)
            all_outputs.extend(probs.cpu().numpy())
    test_loss = total_loss / len(test_loader.dataset)
    
    if verbose:
        print(f"Test loss: {test_loss:.4f}")
        compute_metrics(all_targets, all_preds, all_outputs, class_names=class_names)
    
    metrics = compute_metrics_dict(all_targets, all_preds, all_outputs, class_names)
    metrics['test_loss'] = test_loss
    return {'method': method_name, 'metrics': metrics}



def plot_training_history(history, method_name, test_acc=None, verbose=True):
    """
    Plot training history: train loss, val loss, val/test accuracy.
    
    Args:
        history: dict with keys 'train_loss', 'val_loss', 'val_acc'
        method_name: method name (for plot title)
        test_acc: optional test accuracy value (to show as horizontal line)
        verbose: whether to display plots
    """
    if not verbose:
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss (train and val)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training and Validation Loss - {method_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy (val, test)
    ax2.plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy', linewidth=2)
    if test_acc is not None:
        # Show test accuracy as horizontal line
        ax2.axhline(y=test_acc, color='r', linestyle='--', linewidth=2, label=f'Test Accuracy ({test_acc:.4f})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Accuracy - {method_name}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def print_class_distribution(df, classes_original, title="Class distribution"):
    """Print class distribution."""
    class_counts = df['class'].value_counts().sort_index()
    print(f"\n{title}:")
    for cls_idx, count in class_counts.items():
        if cls_idx < len(classes_original):
            class_name = classes_original[cls_idx]
        else:
            class_name = f"class_{cls_idx}"
        print(f"  Class {cls_idx} ({class_name}): {count} samples")


def print_experiment_results(results, balance_method_list, weight_initialization_list, learning_rate_list=None, batch_size_list=None, momentum_list=None, top_n=5):
    """
    Print top N experiment results in a formatted table.
    Only shows columns for parameters that have more than 1 unique value.

    Args:
        results: List of dicts with experiment results (must include 'balance_method', 'f1_macro', 'test_acc')
        balance_method_list: List of balance methods tested (None, 'oversample', 'class_weight', 'weighted_sampler')
        weight_initialization_list: List of weight_initialization values tested
        learning_rate_list, batch_size_list, momentum_list: optional lists for grid dimensions
        top_n: Number of top results to display (default: 5)
    """
    results_df = pd.DataFrame(results)
    top_results = results_df.nlargest(top_n, 'f1_macro')
    columns_to_show = []
    column_widths = {}

    if len(set(str(m) for m in balance_method_list)) > 1:
        columns_to_show.append(('balance_method', 'Balance_Method', 16))
    if learning_rate_list is not None and len(set(learning_rate_list)) > 1:
        columns_to_show.append(('learning_rate', 'Learning_Rate', 13))
    if batch_size_list is not None and len(set(batch_size_list)) > 1:
        columns_to_show.append(('batch_size', 'Batch_Size', 11))
    if momentum_list is not None and len(set(momentum_list)) > 1:
        columns_to_show.append(('momentum', 'Momentum', 9))
    
    # For weight_initialization, we need to check unique values more carefully
    # Use id() to distinguish different function objects (including lambdas)
    def get_init_id(w):
        if w is None:
            return None
        elif callable(w):
            # Use id to distinguish different function objects
            return id(w)
        else:
            return w
    
    unique_inits = set(get_init_id(w) for w in weight_initialization_list)
    if len(unique_inits) > 1:
        columns_to_show.append(('weight_initialization', 'Weight_Init', 20))
        
        # Create a mapping from function to display name for better readability
        init_name_map = {}
        for idx, w in enumerate(weight_initialization_list):
            if w is None:
                init_name_map[None] = "None"
            elif callable(w):
                if hasattr(w, '__name__') and w.__name__ != '<lambda>':
                    init_name_map[id(w)] = w.__name__
                else:
                    # For lambdas, try to extract from source or use index
                    # Check if it's a lambda that calls init_weights
                    try:
                        import inspect
                        source = inspect.getsource(w)
                        # Try to extract init_type from lambda source
                        if 'init_type=' in source:
                            # Extract the init_type value
                            start = source.find("init_type='") + len("init_type='")
                            end = source.find("'", start)
                            if end > start:
                                init_type = source[start:end]
                                init_name_map[id(w)] = f"init_{init_type}"
                            else:
                                init_name_map[id(w)] = f"init_{idx}"
                        else:
                            init_name_map[id(w)] = f"init_{idx}"
                    except:
                        init_name_map[id(w)] = f"init_{idx}"
            else:
                init_name_map[w] = str(w)
    else:
        init_name_map = {}
    
    # Always show metrics
    columns_to_show.append(('f1_macro', 'F1_Macro', 10))
    columns_to_show.append(('test_acc', 'Test_Acc', 10))
    
    # Calculate total width
    total_width = 6 + sum(width for _, _, width in columns_to_show) + (len(columns_to_show) - 1) * 2
    total_width = max(total_width, 90)
    
    # Print header
    print(f"\n{'='*total_width}")
    print(f"Top {top_n} Best Configurations:")
    print(f"{'='*total_width}")
    
    # Print column headers
    header_parts = ['Rank']
    for _, col_name, width in columns_to_show:
        header_parts.append(f"{col_name:<{width}}")
    print(" ".join(header_parts))
    print("-" * total_width)
    
    # Print rows
    for idx, (_, row) in enumerate(top_results.iterrows(), 1):
        row_parts = [f"{idx:<6}"]
        
        for col_key, _, width in columns_to_show:
            if col_key == 'balance_method':
                row_parts.append(f"{str(row[col_key]):<{width}}")
            elif col_key == 'learning_rate':
                row_parts.append(f"{row[col_key]:<{width}.4f}")
            elif col_key == 'batch_size':
                row_parts.append(f"{row[col_key]:<{width}}")
            elif col_key == 'momentum':
                row_parts.append(f"{row[col_key]:<{width}.3f}")
            elif col_key == 'weight_initialization':
                # Handle weight_initialization - convert to string representation
                val = row[col_key]
                if val is None:
                    val_str = init_name_map.get(None, "None")
                elif callable(val):
                    # Use the mapping we created earlier
                    val_id = id(val)
                    val_str = init_name_map.get(val_id, "lambda")
                else:
                    val_str = init_name_map.get(val, str(val))
                # Truncate if too long
                if len(val_str) > width - 2:
                    val_str = val_str[:width-3] + "..."
                row_parts.append(f"{val_str:<{width}}")
            elif col_key in ['f1_macro', 'test_acc']:
                row_parts.append(f"{row[col_key]:<{width}.4f}")
            else:
                row_parts.append(f"{str(row[col_key]):<{width}}")
        
        print(" ".join(row_parts))
    
    print(f"{'='*total_width}")


def register_activation_and_gradient_saving_hooks(model: nn.Sequential, experiment: Experiment):
    hook_handles = []
    handle = model[0].register_forward_hook(create_input_saving_hook(0, experiment))
    hook_handles.append(handle)
    for i in range(len(model)):
        handle = model[i].register_forward_hook(create_activation_saving_hook(i, experiment))
        hook_handles.append(handle)
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            handle = module.weight.register_hook(
                create_mean_gradient_magnitude_saving_hook(f'layer_{layer_idx}_weight_gradient_magnitudes', experiment)
            )
            hook_handles.append(handle)
            if module.bias is not None:
                handle = module.bias.register_hook(
                    create_mean_gradient_magnitude_saving_hook(f'layer_{layer_idx}_bias_gradient_magnitudes', experiment)
                )
                hook_handles.append(handle)
            layer_idx += 1
    return hook_handles


def get_parameters_as_dict(model):
    params = {}
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            params[f'layer_{layer_idx}_weight'] = module.weight.detach().cpu().numpy().copy()
            if module.bias is not None:
                params[f'layer_{layer_idx}_bias'] = module.bias.detach().cpu().numpy().copy()
            layer_idx += 1
    return params


def save_optimization_step_lengths(old_parameters: Dict[str, np.array], new_parameters: Dict[str, np.array], experiment: Experiment, is_whole_epoch=False):
    for k in old_parameters.keys():
        optimization_step_length = np.linalg.norm(new_parameters[k] - old_parameters[k])
        if is_whole_epoch:
            experiment.save_npy_array(f'{k}_optimization_step_length.npy', optimization_step_length)
        else:
            experiment.save_npy_array(f'batch_{experiment.batch}/{k}_optimization_step_length.npy', optimization_step_length)


def create_activation_saving_hook(layer_idx: int, experiment: Experiment):
    def hook(module, input, output):
        activations = output.detach().cpu().numpy()
        experiment.save_npy_array(f'batch_{experiment.batch}/layer_{layer_idx}_activations.npy', activations)
    return hook


def create_input_saving_hook(layer_idx: int, experiment: Experiment):
    def hook(module, input, output):
        inputs = input[0].detach().cpu().numpy()
        experiment.save_npy_array(f'batch_{experiment.batch}/layer_{layer_idx}_inputs.npy', inputs)
    return hook


def create_mean_gradient_magnitude_saving_hook(key: int, experiment: Experiment):
    def hook(grad):
        experiment.save_npy_array(f'batch_{experiment.batch}/{key}.npy', np.linalg.norm(grad.detach().cpu().numpy()))
        return None  # Don't modify the gradient
    return hook


########################################################
## Training logging helpers (to keep training loop clean):

def setup_epoch_logging(model, experiment):
    """Setup hooks and parameter tracking for verbose epoch logging.
    
    Returns:
        tuple: (hook_handles, parameters_from_epoch_start, old_parameters)
    """
    hook_handles = register_activation_and_gradient_saving_hooks(model, experiment)
    parameters_from_epoch_start = get_parameters_as_dict(model)
    old_parameters = parameters_from_epoch_start
    return hook_handles, parameters_from_epoch_start, old_parameters


def log_batch_step(old_parameters, model, experiment):
    """Log batch-level data and return updated parameters."""
    new_parameters = get_parameters_as_dict(model)
    save_optimization_step_lengths(old_parameters, new_parameters, experiment)
    experiment.next_batch()
    return new_parameters


def finalize_epoch_logging(hook_handles, parameters_from_epoch_start, new_parameters, experiment):
    """Cleanup hooks and save epoch-level optimization step lengths."""
    for handle in hook_handles:
        handle.remove()
    save_optimization_step_lengths(parameters_from_epoch_start, new_parameters, experiment, is_whole_epoch=True)


def log_epoch_metrics(experiment, train_loss, val_loss, val_acc, model):
    """Save epoch metrics and model to experiment."""
    experiment.save_metadata_entry('train_loss', train_loss)
    experiment.save_metadata_entry('val_loss', val_loss)
    experiment.save_metadata_entry('val_acc', val_acc)
    experiment.save_torch_model_sequential('model', model)
    experiment.next_step()


def create_experiment_with_config(learning_rate, momentum, batch_size, weight_initialization):
    """Create an Experiment instance and save configuration."""
    experiment = Experiment()
    experiment.save_metadata_entry(
        'config',
        {
            'learning_rate': learning_rate,
            'momentum': momentum,
            'batch_size': batch_size,
            'weight_initialization': weight_initialization.__name__ if weight_initialization is not None else None
        }
    )
    return experiment


def finalize_training(model, history, test_loader, criterion, class_names, label, verbose):
    """Evaluate model on test set, plot history, and print final metrics.
    
    Returns:
        dict with evaluation results
    """
    result = evaluate_model_on_test(model, test_loader, criterion, class_names, label, verbose)
    test_acc = result['metrics']['accuracy']
    plot_training_history(history, label, test_acc=test_acc, verbose=verbose)
    
    f1_macro = result['metrics']['f1_macro']
    print(f"Test Acc: {test_acc:.4f} | F1 Score: {f1_macro:.4f}")
    
    return result
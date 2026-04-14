import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from helpers.training_utils import get_device

def train_model_with_gradient_tracking_and_accuracy(
    model, 
    train_loader, 
    compute_gradient_norm_fn,
    epochs=15,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
):
    """
    Train model and track gradient norms and training accuracy per epoch.
    
    Parameters:
    -----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        Training data loader
    compute_gradient_norm_fn : callable
        Function that computes the L2 norm of all gradients in the model.
        Should accept a model as parameter and return a float.
    epochs : int, default=15
        Number of training epochs
    lr : float, default=0.01
        Learning rate
    momentum : float, default=0.9
        SGD momentum
    weight_decay : float, default=1e-4
        Weight decay (L2 regularization)
    
    Returns:
    --------
    tuple : (gradient_norms, train_accuracies)
        gradient_norms : list of floats, average gradient norm per epoch
        train_accuracies : list of floats, training accuracy per epoch
    """
    device = get_device()
    model.to(device)
    
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    gradient_norms = []
    train_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_grad_norms = []
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            # Compute gradient norm before optimizer step
            grad_norm = compute_gradient_norm_fn(model)
            epoch_grad_norms.append(grad_norm)
            
            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            opt.step()
        
        # Average gradient norm for this epoch
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        gradient_norms.append(avg_grad_norm)
        
        # Training accuracy for this epoch
        train_acc = correct / total
        train_accuracies.append(train_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Avg Gradient Norm: {avg_grad_norm:.6f}, Train Acc: {train_acc:.4f}")
    
    return gradient_norms, train_accuracies


def compare_models_with_and_without_batch_norm(
    model_fn, 
    name, 
    train_loader, 
    compute_gradient_norm_fn
):
    """
    Compare model with and without BatchNorm by training both variants.
    
    Parameters:
    -----------
    model_fn : callable
        Function that creates a model instance. Should accept use_batch_norm parameter.
    name : str
        Name of the model for logging purposes
    train_loader : DataLoader
        Training data loader
    compute_gradient_norm_fn : callable
        Function that computes the L2 norm of all gradients in the model.
        Should accept a model as parameter and return a float.
    
    Returns:
    --------
    tuple : ((grad_norms_bn, acc_bn), (grad_norms_no, acc_no))
        Two tuples containing gradient norms and accuracies for models with and without BN
    """
    model_bn = model_fn(use_batch_norm=True)
    model_no = model_fn(use_batch_norm=False)

    print(f"\nTraining {name} with BatchNorm...")
    g_bn, acc_bn = train_model_with_gradient_tracking_and_accuracy(
        model_bn, train_loader, compute_gradient_norm_fn
    )
    
    print(f"\nTraining {name} without BatchNorm...")
    g_no, acc_no = train_model_with_gradient_tracking_and_accuracy(
        model_no, train_loader, compute_gradient_norm_fn
    )

    return (g_bn, acc_bn), (g_no, acc_no)


def visualize_batch_norm_ablation_comparison(results, models):
    """
    Create visualization comparing gradient norms and training accuracies
    for models with and without BatchNorm.
    
    Parameters:
    -----------
    results : dict
        Dictionary with model names as keys and nested dictionaries containing:
        - "with_bn": {"grad_norms": list, "accuracies": list}
        - "without_bn": {"grad_norms": list, "accuracies": list}
    models : list of tuples
        List of (model_fn, name) tuples used to generate the results
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    markers = ['o', 's', '^', 'D']
    colors_bn = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_no = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
    
    # Plot gradient norms
    for idx, (model_fn, name) in enumerate(models):
        g_bn = results[name]["with_bn"]["grad_norms"]
        g_no = results[name]["without_bn"]["grad_norms"]
        
        ax1.plot(g_bn, label=f"{name} + BN", marker=markers[idx], 
                color=colors_bn[idx], linewidth=2, markersize=6)
        ax1.plot(g_no, label=f"{name} no BN", marker=markers[idx], 
                color=colors_no[idx], linewidth=2, markersize=6, linestyle='--')
    
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Average Gradient Norm", fontsize=12)
    ax1.legend(fontsize=9, loc='best')
    ax1.set_title("BN effect on gradient norms: All Models Comparison", fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Plot training accuracies
    for idx, (model_fn, name) in enumerate(models):
        acc_bn = results[name]["with_bn"]["accuracies"]
        acc_no = results[name]["without_bn"]["accuracies"]
        
        ax2.plot(acc_bn, label=f"{name} + BN", marker=markers[idx], 
                color=colors_bn[idx], linewidth=2, markersize=6)
        ax2.plot(acc_no, label=f"{name} no BN", marker=markers[idx], 
                color=colors_no[idx], linewidth=2, markersize=6, linestyle='--')
        
        # Add final accuracy text annotation
        final_acc_bn = acc_bn[-1]
        final_acc_no = acc_no[-1]
        ax2.text(len(acc_bn) - 1, final_acc_bn, f' {final_acc_bn:.3f}', 
                verticalalignment='center', fontsize=8, color=colors_bn[idx], fontweight='bold')
        ax2.text(len(acc_no) - 1, final_acc_no, f' {final_acc_no:.3f}', 
                verticalalignment='center', fontsize=8, color=colors_no[idx], fontweight='bold')
    
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Training Accuracy", fontsize=12)
    ax2.legend(fontsize=9, loc='best')
    ax2.set_title("Training Accuracy Curves: All Models Comparison", fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add summary text box with final accuracies
    summary_text = "Final Training Accuracy:\n"
    for model_fn, name in models:
        final_acc_bn = results[name]["with_bn"]["accuracies"][-1]
        final_acc_no = results[name]["without_bn"]["accuracies"][-1]
        summary_text += f"{name} + BN: {final_acc_bn:.4f}\n"
        summary_text += f"{name} no BN: {final_acc_no:.4f}\n"
    
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


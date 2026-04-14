import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def visualize_experiment(histories, input_shape=(1, 3, 32, 32)):
    """
    Show comparison figure (training curves + weight stats).
    Graphviz / torchview-based model structure visualization removed.
    """

    model_names = list(histories.keys())
    assert len(model_names) == 2, "Expected exactly two models"

    name_a, name_b = model_names
    model_a = histories[name_a]["model"][0]
    model_b = histories[name_b]["model"][0]
    hist_a = histories[name_a]["history"][0]
    hist_b = histories[name_b]["history"][0]

    # ==================================================
    # FIGURE 1 — COMPARISON (training behavior)
    # ==================================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # ---- Accuracy ----
    axes[0].plot(hist_a["train_acc"], label=f"{name_a} Train")
    axes[0].plot(hist_a["val_acc"], "--", label=f"{name_a} Val")
    axes[0].plot(hist_b["train_acc"], label=f"{name_b} Train")
    axes[0].plot(hist_b["val_acc"], "--", label=f"{name_b} Val")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy comparison")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ---- Loss ----
    axes[1].plot(hist_a["train_loss"], label=f"{name_a} Train")
    axes[1].plot(hist_a["val_loss"], "--", label=f"{name_a} Val")
    axes[1].plot(hist_b["train_loss"], label=f"{name_b} Train")
    axes[1].plot(hist_b["val_loss"], "--", label=f"{name_b} Val")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss comparison")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # ---- Weight statistics ----
    def weight_stats(model):
        w = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
        return w.mean().item(), w.var().item()

    mean_a, var_a = weight_stats(model_a)
    mean_b, var_b = weight_stats(model_b)

    axes[2].bar(
        [0, 1], [mean_a, mean_b],
        tick_label=[name_a, name_b],
        label="Mean"
    )
    axes[2].bar(
        [0, 1], [var_a, var_b],
        bottom=[mean_a, mean_b],
        alpha=0.6,
        label="Variance"
    )
    axes[2].set_title("Weight statistics")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[2].set_xlabel("Epoch")

    fig.suptitle("Model comparison: training dynamics", fontsize=16)
    plt.show()


def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    return device


def evaluate(model, loader, criterion, device):
    """
    Evaluation for both single-output models and models with auxiliary heads.
    Accuracy is computed using the final classifier output.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)

            # Compute loss (criterion must handle single or multi output)
            loss = criterion(outputs, y)
            loss_value = loss.item()

            # Select final output for accuracy
            if isinstance(outputs, (list, tuple)):
                final_output = outputs[-1]
            else:
                final_output = outputs

            total_loss += loss_value * x.size(0)
            correct += (final_output.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def batch_metrics(x, outputs, loss_value, y):
    """
    Compute batch loss sum and accuracy for both:
    - single-output models
    - multi-output (auxiliary heads) models

    Parameters:
    - outputs: tensor OR list of tensors
    - loss_value: scalar loss tensor (e.g. from CrossEntropyLoss with default mean reduction)
    """
    batch_size = x.size(0)

    # Handle single-output vs multi-output
    if isinstance(outputs, (list, tuple)):
        final_output = outputs[-1]
    else:
        final_output = outputs

    loss_sum = loss_value * batch_size
    correct = (final_output.argmax(1) == y).sum().item()
    total = y.size(0)

    return loss_sum, correct, total


def visualize_extremes(train_epds, val_epds, train_imgs, val_imgs):
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    idx_best_t = train_epds.index(min(train_epds))
    idx_worst_t = train_epds.index(max(train_epds))
    idx_best_v = val_epds.index(min(val_epds))
    idx_worst_v = val_epds.index(max(val_epds))

    samples = [
        (train_imgs[idx_best_t], f"Best tEPD = {train_epds[idx_best_t]}"),
        (train_imgs[idx_worst_t], f"Worst tEPD = {train_epds[idx_worst_t]}"),
        (val_imgs[idx_best_v], f"Best vEPD = {val_epds[idx_best_v]}"),
        (val_imgs[idx_worst_v], f"Worst vEPD = {val_epds[idx_worst_v]}"),
    ]

    for ax, (img, title) in zip(axes.flatten(), samples):
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_dataloaders(batch_size=64):
    print("Loading CIFAR-10 dataset...")
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )

    train_set, val_set, test_set = random_split(
         dataset, [40000, 5000, 5000]
  #      dataset, [1000, 5000, 44000]
    )

    print("Dataset loaded")
    return (
        DataLoader(train_set, batch_size, shuffle=True),
        DataLoader(val_set, batch_size, shuffle=False),
        DataLoader(test_set, batch_size, shuffle=False),
    )
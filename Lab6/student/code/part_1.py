# part_1.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from helpers.training_utils import get_device, evaluate, batch_metrics, visualize_experiment, get_dataloaders
from helpers.draw_architecture_helper import print_and_draw_model_structure

"""
Part 1 — Plain CNN vs ResNet.
"""

class ClassifierHead(nn.Module):
    def __init__(self, in_c, num_classes, use_adaptive_pool=True):
        super().__init__()
        pool_layer = nn.AdaptiveAvgPool2d(1) if use_adaptive_pool else nn.MaxPool2d(2)
        self.net = nn.Sequential(
            pool_layer,
            nn.Flatten(),
            nn.Linear(in_c if use_adaptive_pool else in_c*16*16, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        """
        TODO: Create a conv block with conv2d and relu activation. 
        It should have 2 conv2d layers with the same number of channels and padding 1.
        In the part 1 ignore batch norm.
        """
        super().__init__()    
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        

    def forward(self, x):
        return self.net(x)

class BasicCNN(nn.Module):
    def __init__(self):
        """
        TODO: Create a basic CNN with 2 conv blocks and a classifier head.
        """
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_c=3, out_c=3),
            ConvBlock(in_c=3, out_c=3),
            ClassifierHead(in_c=3, num_classes=10)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, c):
        """
        TODO: Create a residual block with a conv block and a skip connection.
        """
        super().__init__()
        self.block = ConvBlock(in_c=c, out_c=c)
        

    def forward(self, x):
        """
        TODO: Pass input through the conv block and add the skip connection.
        """
        return x + self.block(x)

class SimpleResNet(nn.Module):
    def __init__(self):
        """
        TODO: Create a simple ResNet with 2 residual blocks and a classifier head.
        """
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_c=3, out_c=64),
            ResidualBlock(c=64),
            ResidualBlock(c=64),
            ClassifierHead(in_c=64, num_classes=10, use_adaptive_pool=False)
        )

    def forward(self, x):
        """
        TODO: Pass input through the ResNet.
        """
        return self.net(x)


def train_model(model, train_loader, val_loader, lr, criterion, momentum, weight_decay, epochs):
    device = get_device()
    model.to(device)

    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    for epoch in range(epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            opt.step()

            loss_batch, correct_batch, total_batch = batch_metrics(x, output, loss, y)
            loss_sum += loss_batch
            correct += correct_batch
            total += total_batch

        train_loss, train_acc = loss_sum / total, correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss.item())
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    return model, history


# -------------------------
# Entry
# -------------------------
def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    histories = {
        "BasicCNN": {"model": [], "history": []},
        "SimpleResNet": {"model": [], "history": []}
    }

    for model in [BasicCNN(), SimpleResNet()]:
        trained_model, history = train_model(
            model,
            train_loader,
            val_loader,
            lr=0.01,
            criterion=nn.CrossEntropyLoss(),
            momentum=0.9,
            weight_decay=1e-4,
            epochs=8,
        )

        histories[model.__class__.__name__]["model"].append(trained_model)
        histories[model.__class__.__name__]["history"].append(history)
        print_and_draw_model_structure(
            trained_model,
            output_file=f"{model.__class__.__name__}_fx_graph",
            fmt="svg")

    visualize_experiment(histories)


if __name__ == "__main__":
    main()
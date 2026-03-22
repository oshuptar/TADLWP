import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from part_1 import (
    train_model_with_history,
)

def create_basic_model(num_classes=10):
    """
    TODO: Create a simple 2D CNN model with Conv2d, ReLU, MaxPool2d, Flatten, and Linear layers.
    In data each image have 32x32 pixels and 3 channels (RGB) (3,32,32)
    """
    raise NotImplementedError("TODO: Implement the basic model creation")

def create_lenet_model(num_classes=10):
    """
    TODO: Create LeNet architecture with two Conv2d layers, AvgPool2d, and three fully connected layers.
    In data each image have 32x32 pixels and 3 channels (RGB) (3,32,32)
    """
    raise NotImplementedError("TODO: Implement the LeNet model creation")


def create_alexnet_model(num_classes=10):
    """
    TODO: Create AlexNet architecture with multiple Conv2d layers, MaxPool2d, and fully connected layers with Dropout.
    In data each image have 32x32 pixels and 3 channels (RGB) (3,32,32)
    """
    raise NotImplementedError("TODO: Implement the AlexNet model creation")

def create_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])
    train_set = datasets.CIFAR10(
        root="../dataset",
        train=True,
        download=True,
        transform=transform
    )
    test_set = datasets.CIFAR10(
        root="../dataset",
        train=False,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model):
    print("\n--- Training Conv2d Model on CIFAR-10 ---")
    train_loader, test_loader = create_data_loaders(batch_size=64)
    model = train_model_with_history(
        model,
        train_loader,
        val_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        learning_rate=0.001,
        momentum=0.9,
        epochs=10,
        flatten_input=False
    )

if __name__ == "__main__":
    model = create_basic_model(num_classes=10)
    print(f"\nModel architecture:\n{model}")
    train_model(model)

    model = create_lenet_model(num_classes=10)
    print(f"\nModel architecture:\n{model}")
    train_model(model)

    model = create_alexnet_model(num_classes=10)
    print(f"\nModel architecture:\n{model}")
    train_model(model)
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
    input_channels: int = 3; output_channels: int = 32;
    pooled_length: int = 16 * 16
    flattened_length: int = pooled_length * output_channels
    model = nn.Sequential(
        nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding = 1),
        nn.ReLU(),
        # for 2d the stride acts separately on height and width
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=flattened_length, out_features=num_classes)
    )
    return model;

def create_lenet_model(num_classes=10):
    """
    TODO: Create LeNet architecture with two Conv2d layers, AvgPool2d, and three fully connected layers.
    In data each image have 32x32 pixels and 3 channels (RGB) (3,32,32)
    """

    lenet_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
        nn.AvgPool2d(kernel_size=2, stride = 2), # decrease the size twice
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=6*6*16, out_features=120),
        nn.Sigmoid(),
        nn.Linear(in_features=120, out_features=84),
        nn.Sigmoid(),
        nn.Linear(in_features=84, out_features=10)
    );
    return lenet_model


def create_alexnet_model(num_classes=10):
    """
    TODO: Create AlexNet architecture with multiple Conv2d layers, MaxPool2d, and fully connected layers with Dropout.
    In data each image have 32x32 pixels and 3 channels (RGB) (3,32,32)
    """
    # alexnet_model = nn.Sequential(
    #     nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), # 96x6x6
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=3, stride=2), # 2x2
    #     nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), 
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=3, stride=2),
    #     nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
    #     nn.ReLU(),
    #     nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
    #     nn.ReLU(),
    #     nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=3, stride=2),
    #     nn.Flatten(),
    #     nn.Linear(in_features= , out_features=4096),
    #     nn.ReLU(),
    #     nn.
    # )
    alexnet_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), 
        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=256 * 4 * 4, out_features=4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=4096, out_features=num_classes)
    )
    return alexnet_model

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
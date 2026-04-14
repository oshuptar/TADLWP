import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import Subset

def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    return device

def create_model() -> nn.Sequential:
    image_width: int = 28; image_height: int = 28;
    num_filters: int = 60;
    pooled_length: int = 1;
    flattened_input: int = pooled_length * num_filters;
    num_classes: int = 10
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=7, padding=3, stride=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=image_width),
        nn.Flatten(),
        nn.Linear(
            in_features = flattened_input,
            out_features = num_classes
        ))
    return model;

def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval();
    loss, correct, total = 0.0, 0, 0;
    model.to(device);
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device);
            pred = model(X);
            loss += criterion(pred, y).item();
            correct += (pred.argmax(dim = 1) == y).sum().item();
            total += X.size(dim = 0);

    val_loss, val_acc = loss/len(loader), correct/total;
    return val_loss, val_acc

def train_model(model: nn.Module, train_loader, val_loader, criterion,
                learning_rate, momentum, epochs):
    device = get_device();
    model.to(device);

    for module in model.modules():
        if isinstance(module, nn.Linear) is False:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    #filter(lambda p: p.requires_grad == True, model.parameters()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, );
    for epoch in range(epochs):
        model.train(); # do not forget to reset model to training after evaluation!s
        epoch_loss, correct, total = 0.0, 0, 0;
        for X, y in train_loader:
            optimizer.zero_grad();
            X, y = X.to(device), y.to(device);
            pred = model(X);
            loss = criterion(pred, y);
            loss.backward();
            optimizer.step();   
            epoch_loss += loss.item();
            correct += (pred.argmax(dim = 1) == y).sum().item()
            total += X.size(dim = 0)

        val_loss, val_acc = evaluate(model, val_loader, criterion=criterion, device=device);
        train_loss, train_acc = epoch_loss/len(train_loader), correct/total;
        print(f"Epoch: {epoch}. Train Loss: {train_loss}. Val loss: {val_loss}")
        print(f"Train: Acc: {train_acc}. Train Loss: {train_loss}")
        print(f"Val acc: {val_acc}. Val Loss: {val_loss}\n")
            
    return model

def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # FashionMNIST channel statistics
        transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
    ])

    train_dataset = datasets.FashionMNIST(
        root='data', train=True, download=True, transform=transform
    )
    train_dataset = Subset(train_dataset, range(0, len(train_dataset), 10))
    val_dataset = datasets.FashionMNIST(
        root='data', train=False, download=True, transform=transform
    )
    val_dataset = Subset(val_dataset, range(0, len(val_dataset), 10))
    return train_dataset, val_dataset

def calculate_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad);

def create_data_loaders(batch_size: int = 64):
    train_dataset, val_dataset = get_dataset();
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def main():    
    # Set random seed for reproducibility
    train_loader, val_loader = create_data_loaders(batch_size=6)
    
    # Create model
    model = create_model()
    print(f"\nModel architecture:\n{model}")
    print(f"Number of trainable parameters: {calculate_parameters(model=model)}")
    
    for module in model.modules():
        if isinstance(module, nn.Linear) is False:
            for parameter in module.parameters():
                print(parameter.data)

    model = train_model(model,train_loader, 
                                val_loader,
                                criterion=nn.CrossEntropyLoss(),
                                learning_rate=0.0008, 
                                momentum=0.95, 
                                epochs=2)
    print("after training:")
    for module in model.modules():
        if isinstance(module, nn.Linear) is False:
            for parameter in module.parameters():
                print(parameter.data)

if __name__ == "__main__":
    main()

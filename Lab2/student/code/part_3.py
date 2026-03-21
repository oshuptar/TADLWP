import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List
from myImplementation.dense import Dense
from helpers import (
    generate_spiral_dataset, visualize_dataset, generate_pixels_dataset,
    visualize_predictions, plot_training_progress
)
from experiment_logger_dense import ExperimentWithDense

def create_complex_model(input_size: int = 2, hidden_sizes: List[int] = [64, 32, 16, 8], output_size: int = 1) -> nn.Sequential:
    layers = []
    prev_size: int = input_size;
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(in_features=prev_size, out_features=hidden_size));
        layers.append(nn.Tanh())
        prev_size = hidden_size
    # appending last layer:
    layers.append(nn.Linear(in_features=prev_size, out_features=output_size))
    layers.append(nn.Tanh())
    model = nn.Sequential(*layers)
    return model

def train_model(experiment: ExperimentWithDense, model: nn.Module, dataloader: DataLoader, epochs: int = 200):
    loss_history: list[float] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    model.to(device);
    optimizer = optim.SGD(model.parameters(), lr  = 0.01);
    criterion = nn.MSELoss();
    model.train(); # make sure the model is in training mode
    for epoch in range (epochs):
        epoch_loss: float = 0.0;
        for x,y in dataloader:
            x, y = x.to(device), y.to(device);
            optimizer.zero_grad();
            pred = model(x).squeeze(dim = 1);
            loss = criterion(pred, y);
            loss.backward();
            optimizer.step();   
            epoch_loss += loss.item();
    
        
        loss_history.append(epoch_loss / len(dataloader));
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss: .4f}")
            experiment.log_training_step(model, epoch_loss/len(dataloader))

        if epoch % 100 == 99:
            with torch.no_grad():
                x = dataloader.dataset.tensors[0]
                y = dataloader.dataset.tensors[1]
                pred = model(x).squeeze(dim=1)
                visualize_predictions(x.detach(), y.detach(), pred.detach(), f"Mode progress at Epoch {str(epoch)}");

    plot_training_progress(loss_history)
    return loss_history;

def part3():
    """
    Create a more complicated and interesting model using only Dense layers.
    """
    print("\n=== Part 3 ===")

    # Generate complex dataset (spiral)
    X, y = generate_pixels_dataset()

    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Show dataset before training
    print("Visualizing dataset before training...")
    visualize_dataset(X, y, "Dataset Before Training")

    # Create complex model with multiple Dense layers
    model = create_complex_model(input_size=2, hidden_sizes=[64, 32, 16, 8], output_size=1)

    # Initialize Experiment object for the visualization dashboard
    experiment = ExperimentWithDense()
    experiment.set_training_data(X, y)

    # Train model with built-in visualizations
    train_model(experiment, model, dataloader, epochs=400)
    
    print("✅ Simple model training completed!")

    print(f"Training logged to experiment: {experiment.id}")
    print(f"You can visualize it by running: streamlit run network_dashboard.py")


if __name__ == "__main__":
    part3()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Iterable, List
from myImplementation.dense import Dense
from helpers import generate_simple_line_seperated_dataset, evaluate_model, visualize_dataset, visualize_predictions
from experiment_logger_dense import ExperimentWithDense


def create_simple_model(input_size: int = 2, hidden_size: int = 4, output_size: int = 1) -> nn.Sequential:
    model = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size),
                          nn.ReLU(),
                          nn.Linear(in_features=hidden_size, out_features=output_size));
    return model

def train_model(experiment: ExperimentWithDense, model: nn.Module, dataloader: DataLoader, epochs: int = 100):
    optimizer = optim.SGD(model.parameters(), lr=0.01);
    criterion = nn.MSELoss();
    model.train(); # sets model to training model
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad(); # zeroes the gradient, due ot cummulative nature of gradients
            pred = model(x).squeeze(dim = 1); # squeezes the second column dimension, i.e. is the column dim is of size 1 - it removes it
            #print(pred.shape);
            loss = criterion(pred, y);
            loss.backward(); # computes gradients via autograd saves the results in the .grad property of tensors
            optimizer.step(); # step of the optimizer function (SGD step here)
            # optimize(model.parameters(), 0.01);
            epoch_loss += loss.item();
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
            experiment.log_training_step(model, epoch_loss/len(dataloader))

def optimize(parameters: Iterable[torch.nn.Parameter], learning_rate: float):
    for parameter in parameters:
        if parameter.requires_grad:
            with torch.no_grad():
                if parameter.grad is not None:
                    parameter.data = parameter.data - learning_rate * parameter.grad;




def part2():
    """
    Train a simple model with only Dense layers on a basic classification task.
    """
    print("\n=== Part 2 ===")
    torch.manual_seed(42)

    # Generate simple dataset
    X, y = generate_simple_line_seperated_dataset(n_samples=200, seed=42)
    visualize_dataset(X, y, "Dataset Before Training")
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create simple model with only Dense layers
    model = create_simple_model(input_size=2, hidden_size=4, output_size=1)

    # Initialize Experiment object for the visualization dashboard
    experiment = ExperimentWithDense()
    experiment.set_training_data(X, y)

    # Train model
    train_model(experiment, model, dataloader, epochs=100)

    # Evaluate model
    accuracy, predictions = evaluate_model(model, X, y)
    print(f"Final Accuracy: {accuracy:.4f}")
    visualize_predictions(X, y, predictions, "Final Model Predictions")

    print(f"Training logged to experiment: {experiment.id}")
    print(f"You can visualize it by running: streamlit run network_dashboard.py")

    print("✅ Simple model training completed!")

if __name__ == "__main__":
    part2()

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
    """
    Create a complex model with multiple Dense layers.

    TODO: Implement a deep neural network with multiple hidden layers.
    For each hidden_size in hidden_sizes, add:
    - 1 Dense layer from prev_size to hidden_size
    - Tanh activation (nn.Tanh())
    - Update prev_size = hidden_size
    Finally add:
    - 1 Dense layer from prev_size to output_size

    Args:
        input_size: Input feature size
        hidden_sizes: List of hidden layer sizes
        output_size: Output size

    Returns:
        model: PyTorch Sequential model
    """
    layers = []
    raise NotImplementedError("TODO: Implement the neural network with Dense layers, ReLU, and Sigmoid")
    model = nn.Sequential(*layers)
    return model

def train_model(experiment: ExperimentWithDense, model: nn.Module, dataloader: DataLoader, epochs: int = 200):
    """
    Train a model on the given dataset.

    TODO: Create a training loop similar to the one in part 2 but also save the loss so it can be used by plot_training_progress from helpers.py function to plot the loss at the end.
    Use the visualize_predictions(X, y, predictions, "Model Progress at Epoch " + str(epoch)) to visualize how the prediction changes every 100 epochs.
    Add some checks for example: 
        if epoch % 50 == 49:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
            experiment.log_training_step(model, epoch_loss / len(dataloader))
    Also add some checks every other amount of epochs to also plot the predictions compared to correct data.
    Return the losses list at the end of the training (use: losses.append(epoch_loss / len(dataloader)))
    """
    raise NotImplementedError("TODO: Implement the training loop with loss tracking and visualizations")
    plot_training_progress(loss_history)

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

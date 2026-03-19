import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List
from myImplementation.dense import Dense
from helpers import generate_simple_line_seperated_dataset, evaluate_model, visualize_dataset, visualize_predictions
from experiment_logger_dense import ExperimentWithDense


def create_simple_model(input_size: int = 2, hidden_size: int = 4, output_size: int = 1) -> nn.Sequential:
    """
    Create a simple model with Dense layers.
    
    TODO: Implement a neural network with:
    - 1 Dense layer from input_size to hidden_size
    - ReLU activation (nn.ReLU())
    - 1 Dense layer from hidden_size to output_size  
    - No activation function at the end — do NOT add Sigmoid.
        BCEWithLogitsLoss (used later during training) already includes
        a numerically stable sigmoid internally.
    
    Args:
        input_size: Input feature size
        hidden_size: Hidden layer size
        output_size: Output size
        
    Returns:
        model: PyTorch Sequential model
    """
    model = nn.Sequential()
    raise NotImplementedError("TODO: Implement the neural network with Dense layers, ReLU, and Sigmoid")
    return model

def train_model(experiment: ExperimentWithDense, model: nn.Module, dataloader: DataLoader, epochs: int = 100):
    """
    Train a model on the given dataset.
    
    TODO: Implement training loop for a model that will take epochs amount of iterations.
    Use PyTorch's SGD optimizer: optim.SGD(model.parameters(), lr=...) and BCEWithLogitsLoss, print result loss.
    Every 10 epochs:
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
            experiment.log_training_step(model, epoch_loss / len(dataloader))
    and call experiment.log_training_step(model, mean_epoch_loss) to save the results for visualization.

    Hint: Use for X, y in dataloader: to iterate through batches
    """
    raise NotImplementedError("TODO: Implement the training loop with SGD optimizer and BCEWithLogitsLoss")

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

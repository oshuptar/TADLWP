import os
import sys
import torch
import torch.nn as nn
from myImplementation.dense import Dense
from torchviz import make_dot
from PIL import Image

"""
TODO: Implement the forward and backward pass for the Dense layer in the myImplementation/dense.py file
"""

def part1():
    """
    Test the Dense layer implementation with forward and backward propagation.
    """
    print("=== Part 1 ===")
    
    # Create test data with fixed seed for reproducibility
    torch.manual_seed(42)
    batch_size, input_size, output_size = 4, 3, 2
    x = torch.randn(batch_size, input_size, requires_grad=True)
    
    # Create Dense layer
    dense = Dense(input_size, output_size)
    
    # Forward pass
    output = dense(x)
    
    #Saving output (loss) for visualization of autoGrad:
    dot = make_dot(output, params={"weight": dense.weight, "bias": dense.bias})
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    # Simple checks to confirm it works
    assert x.grad is not None, "Input gradients should be computed"
    assert dense.weight.grad is not None, "Weight gradients should be computed"
    assert dense.bias.grad is not None, "Bias gradients should be computed"
    assert not torch.isnan(output).any(), "Output should not contain NaN values"
    assert not torch.isnan(x.grad).any(), "Input gradients should not contain NaN values"
    
    print("All tests passed ✅")
    print(f"Output from forward pass:\n{output}")
    print(f"W gradient in backward pass:\n{dense.weight.grad}")
    
    tmp_path = dot.render('autograd_visualization', format="png")
    if 'google.colab' in sys.modules:
      os.remove(tmp_path.replace(".png", ""))  # removes the .gv file
      print(f"You're on colab. To see autograd, open {tmp_path}")
    else:
      print("Showing autograd:")
      img = Image.open(tmp_path)
      img.show()
      os.remove(tmp_path)
      os.remove(tmp_path.replace(".png", ""))  # also removes the .gv file

if __name__ == "__main__":
    part1()

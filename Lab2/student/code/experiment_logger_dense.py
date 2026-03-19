"""
Helper functions for logging training experiments to the network dashboard.
"""
import torch
import torch.nn as nn
import sys
import os
from typing import Any


from experiment_logger import Experiment
from myImplementation.dense import Dense


class ExperimentWithDense(Experiment):
    """
    Extended Experiment class that can load models with Dense layers.
    """
    def set_training_data(self, X: torch.Tensor, y: torch.Tensor):
        '''Once set, the training data will be added to each experiment's step.
        '''
        self.train_X = X.reshape(-1, X.shape[-1]).numpy()
        y_onehot = torch.zeros(len(y), 2)
        y_onehot[range(len(y)), y.long()] = 1.0
        self.train_y = y_onehot.numpy()

    def load_torch_model_sequential(self, key: str):
        '''Overwriten to make sure the custom Dense layer is loaded correctly.
        '''
        filepath = self._load_key_to_sanitized_filepath(key)
        
        loaded_data = torch.load(filepath)
        layer_definitions = loaded_data['model_architecture']
        
        layers = []
        for layer_info in layer_definitions:
            layer_type_str = layer_info['type']
            layer_params = layer_info['params']
            
            # Try to get layer from nn first (for Linear, ReLU, etc.)
            LayerClass = getattr(nn, layer_type_str, None)
            
            # If not found in nn, try to get Dense from myImplementation
            if LayerClass is None and layer_type_str == 'Dense':
                LayerClass = Dense

            if LayerClass:
                new_layer = LayerClass(**layer_params)
                layers.append(new_layer)
            else:
                raise ValueError(f"Unknown layer type: {layer_type_str}")

        new_model = nn.Sequential(*layers)
        new_model.load_state_dict(loaded_data['model_state_dict'])
        return new_model

    def log_training_step(self, model: Any, loss: float):
        '''Save the model and loss for dashboard visualization.

            Args:
                model: nn.Model or myimplementation.Dense
                loss: Value of the loss function
        '''
        self.save_npy_array('train_X', self.train_X)
        self.save_npy_array('train_y', self.train_y)
        self.save_metadata_entry('train_loss', loss)
        model_for_visualization = nn.Sequential(*list(model.children()))
        self.save_torch_model_sequential('model', model_for_visualization)
        self.next_step()

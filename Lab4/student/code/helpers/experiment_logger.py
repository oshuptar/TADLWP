import os
import json
import torch
import shutil
from typing import Any

import numpy as np
import torch.nn as nn
from wonderwords import RandomWord


class Experiment:
    '''
    A simple container for saving things related to deep network training and usage.
    '''
    def __init__(self, id: str='random', persist_dir: str='./experiments', verbose: bool=True, overwrite: bool=False):
        '''
        Creates a new experiment or loads an existing one.
        '''
        if id == 'random':
            r = RandomWord()
            free_name_found = False
            while not free_name_found:
                random_adjective = r.word(include_categories=["adjective"])
                random_noun = r.word(include_parts_of_speech=["noun"])
                self.id = f'{random_adjective}-{random_noun}'
                if not os.path.isdir(os.path.join(persist_dir, self.id)):
                    free_name_found = True
        else:
            self.id = id
        self.experiment_dir = os.path.join(persist_dir, self.id)
        self.verbose = verbose
        try:
            os.makedirs(self.experiment_dir)
            self.step = 0
            self.batch = 0
            print(f'Experiment {self.id} started!')
        except OSError:
            if overwrite:
                if verbose:
                    print(f'Overwritten existing experiment with name: {self.id}')
                shutil.rmtree(self.experiment_dir)
                os.makedirs(self.experiment_dir)
                self.step = 0
                self.batch = 0
            else:
                if verbose:
                    print(f'Loaded existing experiment with name: {self.id}')
                self.step = self.last_step
                self.batch = 0

    def next_step(self):
        '''
        Increments the step counter.
        '''
        self.step += 1

    @property
    def last_step(self):
        '''
        Returns the last step number from the experiment directory.
        '''
        steps = []
        for p in os.listdir(self.experiment_dir):
            full_path = os.path.join(self.experiment_dir, p)
            if os.path.isdir(full_path):
                try:
                    steps.append(int(p))
                except ValueError:
                    continue
        return max(steps) if steps else 0
    
    def next_batch(self):
        '''
        Increment the batch counter.
        '''
        self.batch += 1

    def save(self, name: str, data: Any, step: int = None):
        '''
        Saves data to a file in the experiment directory.
        
        Args:
            name: Name of the file (without extension)
            data: Data to save (dict, list, tensor, numpy array, etc.)
            step: Step number (defaults to self.step)
        '''
        if step is None:
            step = self.step
        
        step_dir = os.path.join(self.experiment_dir, str(step))
        os.makedirs(step_dir, exist_ok=True)
        
        filepath = os.path.join(step_dir, name)
        
        if isinstance(data, torch.Tensor):
            torch.save(data, filepath)
        elif isinstance(data, np.ndarray):
            torch.save(torch.from_numpy(data), filepath)
        elif isinstance(data, (dict, list)):
            # Save as JSON for dicts and lists
            json_path = filepath if filepath.endswith('.json') else filepath + '.json'
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            # Try to save as pickle
            import pickle
            pickle_path = filepath if filepath.endswith('.pkl') else filepath + '.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)
        
        if self.verbose:
            print(f'Saved {name} to {filepath}')

    def load(self, name: str, step: int = None):
        '''
        Loads data from a file in the experiment directory.
        
        Args:
            name: Name of the file (without extension)
            step: Step number (defaults to self.step)
        
        Returns:
            Loaded data
        '''
        if step is None:
            step = self.step
        
        step_dir = os.path.join(self.experiment_dir, str(step))
        filepath = os.path.join(step_dir, name)
        
        if os.path.exists(filepath):
            return torch.load(filepath)
        elif os.path.exists(filepath.replace('.pt', '.json')):
            with open(filepath.replace('.pt', '.json'), 'r') as f:
                return json.load(f)
        elif os.path.exists(filepath.replace('.pt', '.pkl')):
            import pickle
            with open(filepath.replace('.pt', '.pkl'), 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f'File {filepath} not found')
    
    def _save_key_to_sanitized_filepath(self, key: str) -> str:
        filepath = os.path.join(self.experiment_dir, str(self.step), key)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if os.path.isfile(filepath) and self.verbose:
            print(f'Overwriting existing key: {key}. Make sure you didn\'t want to increment the step first')
        return filepath

    def _load_key_to_sanitized_filepath(self, key: str) -> str:
        filepath = os.path.join(self.experiment_dir, str(self.step), key)
        if not os.path.isfile(filepath) and self.verbose:
            print(f'Key: {key} doesn\'t point to a file. An error will be thrown.')
        return filepath
    
    def save_torch_model_sequential(self, key: str, model: Any):
        filepath = self._save_key_to_sanitized_filepath(key)

        layer_definitions = []
        for name, module in model.named_children():
            layer_info = {
                'type': type(module).__name__,
                'params': {}
            }
            if hasattr(module, 'weight'):
                if len(module.weight.shape) == 2:  # Linear layer
                    layer_info['params']['in_features'] = module.weight.shape[1]
                    layer_info['params']['out_features'] = module.weight.shape[0]
                elif len(module.weight.shape) == 4:  # Conv2d layer
                    layer_info['params']['in_channels'] = module.weight.shape[1]
                    layer_info['params']['out_channels'] = module.weight.shape[0]
                    layer_info['params']['kernel_size'] = (module.weight.shape[2], module.weight.shape[3])
            
            layer_definitions.append(layer_info)

        save_data = {
            'model_state_dict': model.state_dict(),
            'model_architecture': layer_definitions
        }
        torch.save(save_data, filepath)

    def load_torch_model_sequential(self, key: str):
        filepath = self._load_key_to_sanitized_filepath(key)

        loaded_data = torch.load(filepath)
        layer_definitions = loaded_data['model_architecture']

        layers = []
        for layer_info in layer_definitions:
            layer_type_str = layer_info['type']
            layer_params = layer_info['params']
            LayerClass = getattr(nn, layer_type_str, None)

            if LayerClass:
                new_layer = LayerClass(**layer_params)
                layers.append(new_layer)
            else:
                raise ValueError(f"Unknown layer type: {layer_type_str}")
        new_model = nn.Sequential(*layers)
        new_model.load_state_dict(loaded_data['model_state_dict'])
        return new_model

    def save_npy_array(self, key: str, value: np.ndarray):
        filepath = self._save_key_to_sanitized_filepath(key)
        np.save(filepath, value)

    def load_npy_array(self, key: str) -> np.ndarray:
        filepath = self._load_key_to_sanitized_filepath(key)
        return np.load(filepath)

    def save_metadata_entry(self, key: str, value: Any):
        '''
        The entry must be json-serializable
        '''
        filepath = os.path.join(self.experiment_dir, str(self.step), '_metadata.json')
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        if key in metadata and self.verbose:
            print(f'Overwriting existing key: {key}. Make sure you didn\'t want to increment the step first')
        metadata[key] = value
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metadata, f)

    def load_metadata_entry(self, key: str):
        filepath = self._load_key_to_sanitized_filepath('_metadata.json')
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        return metadata[key]

    def load_metadata_entry_history(self, key: str):
        current_step = self.step
        for step in range(0, self.last_step+1):
            self.step = step
            filepath = self._load_key_to_sanitized_filepath('_metadata.json')
            with open(filepath, 'r') as f:
                metadata = json.load(f)
            yield metadata[key]
        self.step = current_step


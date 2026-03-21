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
            print(f'Experiment {self.id} started!')
        except OSError:
            if overwrite:
                if verbose:
                    print(f'Overwritten existing experiment with name: {self.id}')
                shutil.rmtree(self.experiment_dir)
                os.makedirs(self.experiment_dir)
                self.step = 0
            else:
                if verbose:
                    print(f'Loaded existing experiment with name: {self.id}')
                self.step = self.last_step
        self.batch = 0

    def next_step(self):
        '''
        Increment the step. Every step's data is saved in a separate directory.
        '''
        self.step += 1
        self.batch = 0

    def next_batch(self):
        '''
        Increment the batch. Used by some handlers.
        '''
        self.batch += 1

    @property
    def last_step(self):
        steps = []
        for p in os.listdir(self.experiment_dir):
            full_path = os.path.join(self.experiment_dir, p)
            if os.path.isdir(full_path):
                steps.append(int(p))
        return max(steps)

    @property
    def last_batch(self):
        batches = []
        for p in os.listdir(f'{self.experiment_dir}/{self.step}'):
            if not p.startswith('batch_'):
                continue
            batches.append(int(p.replace('batch_', '')))
        return max(batches)

    @property
    def disk_size(self):
        '''
        Size of all files saved to disk for this experiment, in Megabytes
        '''
        size_in_bytes = get_dir_size_in_bytes(self.experiment_dir)
        return size_in_bytes / 1_048_576

    def _save_key_to_sanitized_filepath(self, key: str) -> str:
        filepath = os.path.join(self.experiment_dir, str(self.step), key)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if os.path.isfile(filepath) and self.verbose:
            print(f'Overwriting existing key: {key}. Make sure you didn\'t want to increment the step first')
        return filepath

    def _load_key_to_sanitized_filepath(self, key: str) -> str:
        filepath = os.path.join(self.experiment_dir, str(self.step), key)
        if not os.path.isfile(filepath) and self.verbose:
            print(f'Key: {key} doesn\t point to a file. An error will be thrown.')
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
                layer_info['params']['in_features'] = module.weight.shape[1]
                layer_info['params']['out_features'] = module.weight.shape[0]

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


class ExperimentRegistry:
    def __init__(self, persist_dir: str='./experiments', verbose: bool=True):
        self.persist_dir = persist_dir
        self.verbose = verbose

    def get_experiment_names(self):
        names = []
        for experiment_name in os.listdir(self.persist_dir):
            experiment_path = os.path.join(self.persist_dir, experiment_name)
            if os.path.isfile(experiment_path):
                continue
            if os.path.isdir(experiment_path):
                names.append(experiment_name)
        names = sorted(names, key=lambda x: -os.path.getmtime(os.path.join(self.persist_dir, x)))
        return names

    def remove_smaller_than(self, disk_size_in_mb: float):
        '''
        Remove experiments that have disk size in Megabytes lower than the threshold
        '''
        for experiment_id in os.listdir(self.persist_dir):
            experiment_path = os.path.join(self.persist_dir, experiment_id)
            if os.path.isfile(experiment_path):
                continue
            if os.path.isdir(experiment_path):
                experiment_disk_size = get_dir_size_in_bytes(experiment_path) / 1_048_576
                if experiment_disk_size < disk_size_in_mb:
                    if self.verbose:
                        print(f'Removing experiment {experiment_id}')
                    shutil.rmtree(experiment_path)

    def remove_shorter_than(self, num_steps: int):
        '''
        Remove experiments that have fewer steps than the threshold
        '''
        for experiment_id in os.listdir(self.persist_dir):
            experiment_path = os.path.join(self.persist_dir, experiment_id)
            if os.path.isfile(experiment_path):
                continue
            if os.path.isdir(experiment_path):
                experiment_num_steps = 0
                for step in os.listdir(experiment_path):
                    step_path = os.path.join(experiment_path, step)
                    if os.path.isdir(step_path):
                        experiment_num_steps += 1
                if experiment_num_steps < num_steps:
                    if self.verbose:
                        print(f'Removing experiment {experiment_id}')
                    shutil.rmtree(experiment_path)

# remove_older_than

def get_dir_size_in_bytes(path):
    total = 0
    for p in os.listdir(path):
        full_path = os.path.join(path, p)
        if os.path.isfile(full_path):
            total += os.path.getsize(full_path)
        elif os.path.isdir(full_path):
            total += get_dir_size_in_bytes(full_path)
    return total

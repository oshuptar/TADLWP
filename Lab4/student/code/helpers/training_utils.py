import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def divide_data_to_x_and_y(df):
    x = df.drop('class', axis=1).values
    y = df['class'].values
    return x, y


def divide_data_to_train_val_test(df, test_size=0.3, val_size=0.1):
    x, y = divide_data_to_x_and_y(df)
    train_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        x, y, test_size=(1.0 - train_size), stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(test_size/(test_size + val_size)), stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_coffee_dataset(dataset_dir='dataset/Coffee'):
    """
    Load the Coffee time series dataset from .ts files.
    
    The Coffee dataset is a binary classification problem to distinguish between
    Robusta and Arabica coffee beans using Fourier Transform Infrared Spectroscopy.
    
    Args:
        dataset_dir: Directory containing the Coffee dataset files
    
    Returns:
        X_train: numpy array of shape (num_train_samples, 1, sequence_length) - training 1D signals
        y_train: numpy array of shape (num_train_samples,) - training class labels
        X_test: numpy array of shape (num_test_samples, 1, sequence_length) - test 1D signals
        y_test: numpy array of shape (num_test_samples,) - test class labels
        sequence_length: Length of each time series (286 for Coffee dataset)
    """
    import os
    
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, dataset_dir, 'Coffee_TRAIN.ts')
    test_file = os.path.join(current_dir, dataset_dir, 'Coffee_TEST.ts')
    
    def parse_ts_file(filepath):
        """Parse a .ts file and extract time series data and labels."""
        X = []
        y = []
        sequence_length = None
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # Skip header lines (starting with @ or #)
            data_started = False
            for line in lines:
                line = line.strip()
                
                # Parse header information
                if line.startswith('@seriesLength'):
                    sequence_length = int(line.split()[1])
                
                # Start reading data after @data
                if line == '@data':
                    data_started = True
                    continue
                
                if data_started and line and not line.startswith('@') and not line.startswith('#'):
                    # Parse data line: comma-separated values followed by :class_label
                    if ':' in line:
                        parts = line.rsplit(':', 1)
                        values_str = parts[0]
                        label = int(parts[1])
                        
                        # Parse comma-separated values
                        values = [float(x) for x in values_str.split(',')]
                        
                        # Convert to numpy array and reshape to (1, sequence_length) for Conv1d
                        signal = np.array(values, dtype=np.float32)
                        X.append(signal.reshape(1, -1))
                        y.append(label)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        return X, y, sequence_length
    
    # Load training and test data
    X_train, y_train, sequence_length = parse_ts_file(train_file)
    X_test, y_test, _ = parse_ts_file(test_file)
    
    # Normalize each signal (zero mean, unit variance)
    for i in range(len(X_train)):
        signal = X_train[i, 0, :]
        mean = signal.mean()
        std = signal.std()
        if std > 1e-8:
            X_train[i, 0, :] = (signal - mean) / std
    
    for i in range(len(X_test)):
        signal = X_test[i, 0, :]
        mean = signal.mean()
        std = signal.std()
        if std > 1e-8:
            X_test[i, 0, :] = (signal - mean) / std
    
    return X_train, y_train, X_test, y_test, sequence_length

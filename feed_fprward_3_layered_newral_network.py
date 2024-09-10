import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the MLP network
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, lambda_reg):
        super(MLP, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, hidden_size_1)
        self.hidden_layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.output_layer = nn.Linear(hidden_size_2, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation in hidden layers
        self.lambda_reg = lambda_reg

    def forward(self, x):
        x = self.sigmoid(self.hidden_layer_1(x))
        x = self.sigmoid(self.hidden_layer_2(x))
        x = self.output_layer(x)  # Linear activation for the output
        return x

# Generate sliding windows from time series
def create_time_series_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

# Define the early stopping function
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, validation_loss):
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = validation_loss
            self.counter = 0

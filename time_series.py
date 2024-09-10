import matplotlib.pyplot as plt
import numpy as np
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
        
        self.criterion = nn.MSELoss()  # Loss function for time series (mean squared error)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=lambda_reg)
        self.early_stopping = EarlyStopping(patience=patience)


    def forward(self, x):
        x = self.sigmoid(self.hidden_layer_1(x))
        x = self.sigmoid(self.hidden_layer_2(x))
        x = self.output_layer(x)  # Linear activation for the output
        return x
    def fit(self, train_loader, val_loader, batch_size = 40 ,epochs = 1000):
        train_error = []
        val_error = []
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            val_loss = self.predict(val_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.6f}, '
            f'Val Loss: {val_loss / len(val_loader):.6f}')
            if self.early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch + 1}')
                break
            train_error.append(train_loss/len(train_loader))
            val_error.append(val_loss)
        return train_error, val_error
    def predict(self, val_loader):
        val_loss = 0.0
        self.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
            return val_loss / len(val_loader)

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
def eulers_method(seq_length=2000, beta=0.2, gamma=0.1, tau=25,
    x0=1.5,sigma = 0.15, n=10):
    x = np.zeros((seq_length, 1))
    x[0] = x0
    for t in range(0, seq_length-1):
        x[t+1] = x[t] + beta * x[t - tau] / (1 + x[t - tau]**10) - gamma * x[t]
    #noise = np.random.normal(0, sigma, size=x.shape)  # Generate noise
    inputs = []
    outputs = []
    plt.plot(x[:1500])
    plt.show()
    x += np.random.normal(0, 1, x.shape)

    for t in range(300, 1500):
        # Input consists of the values at t-20, t-15, t-10, t-5, and t
        input_values = [x[t-20], x[t-15], x[t-10], x[t-5], x[t]]
        
        # Output is the value at t+5
        output_value = x[t+5]
        
        inputs.append(input_values)
        outputs.append(output_value)
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    inputs = inputs.reshape(inputs.shape[0], -1)
    return torch.Tensor(inputs), torch.Tensor(outputs)


inputs, outputs = eulers_method()
plt.plot()

x_train, x_val, x_test, y_train, y_val, y_test = inputs[:800,:], inputs[800:1000,:], inputs[1000:,:], outputs[:800,:], outputs[800:1000,:], outputs[1000:,:]



print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

batch_size = 16
lambda_reg = 1e-4  # Weight decay for regularization
learning_rate = 0.01
patience = 10

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = x_train.shape[1]
hidden_size_1 = 5
hidden_size_2 = 5
output_size = 1
lambda_reg = 1e-4  # Weight decay for regularization
learning_rate = 0.001
epochs = 100
patience = 100  # Early stopping patience

# Initialize the MLP model

    
batch_size = 32
epochs = 10000

train_samples =  len(y_train)
test_samples =  len(y_test)

model = MLP(input_size, hidden_size_1, hidden_size_2, output_size, lambda_reg)


val_error, train_error = model.fit(train_loader, val_loader)
test_loss = model.predict(test_loader)
print(f'Test Loss: {test_loss:.6f}')
plt.plot(train_error)
plt.plot(val_error)
plt.show()
#train_dataset, test_dataset = create_dataset_20_80_from_classA(classA, classB)


plt.show()
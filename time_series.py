import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset
import math
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
            val_loss = self.predict(val_loader)[1]
            #print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.6f}, '
            #f'Val Loss: {val_loss / len(val_loader):.6f}')
            #if self.early_stopping.early_stop:
            #    print(f'Early stopping at epoch {epoch + 1}')
            #    break
            train_error.append(train_loss/len(train_loader))
            val_error.append(val_loss)
        return train_error, val_error
    def predict(self, val_loader):
        val_loss = 0.0
        self.eval()
        total_outputs = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.forward(inputs)
                #print(outputs.flatten().tolist())
                total_outputs.extend(outputs.flatten().tolist())
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
            return total_outputs,  val_loss / len(val_loader)

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
    print(x[300:1100,].shape)
    plt.plot(x[:1500])
    plt.show()
    x[300:1100,] += np.random.normal(0, 0.15, (800,1))

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

def plot_output(values, label):
    values = np.array(values)
    mean_values = np.mean(values, axis=0)
    std_dev_values = np.std(values, axis=0)
    plt.plot( mean_values, label=label)
    plt.fill_between( mean_values - std_dev_values, mean_values + std_dev_values, alpha=0.2)#, label='Train Standard Deviation')

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

batch_size = 40
lambda_reg = 1e-3  # Weight decay for regularization
learning_rate = 0.001
patience = 10

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = x_train.shape[1]
hidden_size_1 = 5
hidden_size_2 = 5
output_size = 1
lambda_reg = 0 # 1e-4  # Weight decay for regularization
learning_rate = 0.005

# Initialize the MLP model

    
batch_size = 32
epochs = 2000

train_samples =  len(y_train)
test_samples =  len(y_test)

best_error = math.inf
best_model = None
worst_error  =  0
worst_model = None
trial = 0
best_hyperparameters = {"hidden_size_1":0, "hidden_size_2":0,"lambda_reg":0}#4, 5
worst_hyperparameters = {"hidden_size_1":0, "hidden_size_2":0,"lambda_reg":0}#4, 5

hyperparameters = {"hidden_size_1":[3,5,6], "hidden_size_2":[4, 8,9],"lambda_reg":[0,1e-4,1e-3,1e-2]}#4, 5
for lambda_reg in hyperparameters["lambda_reg"]:
    for hidden_size_1 in hyperparameters["hidden_size_1"]:
        for hidden_size_2 in hyperparameters["hidden_size_2"]:
            model = MLP(input_size, hidden_size_1, hidden_size_2, output_size, lambda_reg)
            train_error, val_error = model.fit(train_loader, val_loader)

            if val_error[-1] < best_error :
                best_error = val_error[-1]
                best_model = model
                best_hyperparameters["hidden_size_1"] = hidden_size_1
                best_hyperparameters["hidden_size_2"] = hidden_size_2
                best_hyperparameters["lambda_reg"] = lambda_reg
            if val_error[-1] > worst_error : 
                worst_error = val_error[-1]
                worst_model = model
                worst_hyperparameters["hidden_size_1"] = hidden_size_1
                worst_hyperparameters["hidden_size_2"] = hidden_size_2
                worst_hyperparameters["lambda_reg"] = lambda_reg
            trial += 1
            print("Running trial : ",trial," hidden size 1 : ",hidden_size_1," hidden size 2 : ",hidden_size_2," lambda_reg : ",lambda_reg," validation error : ",val_error[-1],"train error :",train_error[-1])

#test_loss = best_model.predict(test_loader)
plt.plot(y_test.tolist(),label='function ')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# List of 5 different seed values
seeds = [42, 7, 123, 1000, 2024]

# Loop through each seed, set it, and perform an example task
print("runs for our best models")
predictions = []
for seed in seeds:
    set_seed(seed)
    # Example task: create a random tensor
    # Print seed and the random tensor generated
    print(f"Seed: {seed}")
    model = MLP(input_size,  best_hyperparameters["hidden_size_1"], best_hyperparameters["hidden_size_2"], output_size, best_hyperparameters["lambda_reg"])
    train_error, val_error = model.fit(train_loader, val_loader)
    outputs, loss =  model.predict(test_loader)
    predictions.append(outputs)
    print(f'Best model test Loss: { loss:.6f}')
plot_output(predictions,"best model")

#plt.plot(predictions,label='best model ')
print("runs for our worst models")
predictions = []
for seed in seeds:
    set_seed(seed)
    # Example task: create a random tensor
    # Print seed and the random tensor generated
    print(f"Seed: {seed}")
    model = MLP(input_size,  worst_hyperparameters["hidden_size_1"], worst_hyperparameters["hidden_size_2"], output_size, worst_hyperparameters["lambda_reg"])
    train_error, val_error = model.fit(train_loader, val_loader)
    outputs, loss =  model.predict(test_loader)
    predictions.append(outputs)
    print(f'Worst model test Loss: { loss:.6f}')
plot_output(predictions,"worst model")

print("best hyperparameters : ",best_hyperparameters)
print("worst hyperparameters : ",worst_hyperparameters)

print(f'Worst model test Loss: { loss:.6f}')
#plt.plot(train_error)
#plt.plot(val_error)
plt.legend()
plt.show()
#train_dataset, test_dataset = create_dataset_20_80_from_classA(classA, classB)


plt.show()

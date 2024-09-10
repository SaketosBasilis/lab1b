import numpy as np
import feed_forward_newral_network
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Create the patterns using an identity matrix of size 8, multiplied by 2 and subtract 1
patterns = np.eye(8) * 2 - 1

# Set the targets equal to the patterns
targets = patterns

# Print the result
#print("Patterns:\n", patterns)
#print("Targets:\n", targets)

epochs = 2000

nn = feed_forward_newral_network.NeuralNetwork(input_size=8, hidden_size=3, output_size=8, hta=0.1)
#X_train, X_test, y_train, y_test = train_test_split(patterns, targets, test_size=0.33, random_state=42)
X_train, y_train = patterns, targets
train_samples = len(y_train)
#test_samples = len(y_test)
training_error =  []
test_error =  []

for i in range(epochs):
    epoch_train_error = 0
    epoch_test_error = 0
    X = X_train
    T = y_train
    Y = nn.forward(X)

    #print("Y_train : ",Y)
    #print("T_train : ",T)
    #rint("nn.hidden_layer_output : ",np.where(nn.hidden_layer_output > 0, 1, 0) )

    nn.backward(T)
    
    T= T.T.flatten()
    Y= Y.T.flatten()
    epoch_train_error += np.sum((Y-T)**2)/train_samples
    training_error.append(epoch_train_error)


    #test time
    #X = X_test
    #T = y_test
    #Y = nn.forward(X)
    #epoch_test_error += np.sum((Y-T)**2)/test_samples
    #print("Y_test : ",Y)
    #print("T_test : ",T)
    #test_error.append(epoch_test_error)
Y = nn.forward(X)
    #epoch_test_error += np.sum((Y-T)**2)/test_samples
for i, input in enumerate(X):
    print("input : ",input," hidden : ",np.where(nn.hidden_layer_output[i] > 0, 1, 0))
#print("Y_test : ",Y)
#print("T_test : ",T)
#print("nn.hidden_layer_output : ",np.where(nn.hidden_layer_output > 0, 1, 0) )

plt.plot(training_error)
plt.plot(test_error)
plt.show()

import matplotlib.pyplot as plt
import feed_forward_newral_network
import numpy as np
def eulers_method(seq_length=2000, beta=0.2, gamma=0.1, tau=25,
    x0=1.5, n=10):
    x = np.zeros((seq_length, 1))
    x[0] = x0
    for t in range(0, seq_length-1):
        x[t+1] = x[t] + beta * x[t - tau] / (1 + x[t - tau]**10) - gamma * x[t]
    return x[300:2000]

inputs = []
outputs = []
x = eulers_method()

for t in range(300, 1500):
    # Input consists of the values at t-20, t-15, t-10, t-5, and t
    input_values = [x[t-20], x[t-15], x[t-10], x[t-5], x[t]]
    
    # Output is the value at t+5
    output_value = x[t+5]
    
    inputs.append(input_values)
    outputs.append(output_value)
inputs = np.array(inputs)
outputs = np.array(outputs)
print(inputs.shape)
inputs = inputs.reshape(inputs.shape[0], -1)

print(outputs.shape)
x_train, x_val, x_test, y_train, y_val, y_test = inputs[:800,:], inputs[800:1000,:], inputs[1000:,:], outputs[:800,:], outputs[800:1000,:], outputs[1000:,:]

print(x_train[:5,:])
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


nn = feed_forward_newral_network.NeuralNetwork(input_size=5, hidden_size=2, output_size=1, hta=0.1)
batch_size = 4
epochs = 10000
train_error = []
test_error = []
train_samples =  len(y_train)
test_samples =  len(y_test)
for i in range(epochs):
    epoch_train_error = 0
    for i in range(0,train_samples,batch_size):
        X = x_train[i:i+batch_size,:]
        T = y_train[i:i+batch_size]
        Y = nn.forward(X)
        nn.backward(T)
        Y= Y.T.flatten()
        T = T.flatten()
        #print("Y : ",Y)
        #print("T : ",T)
        
        
        epoch_train_error += np.sum((Y-T)**2)
    epoch_test_error = 0
    for i in range(0,test_samples,batch_size):
        X = x_test[i:i+batch_size,:]
        T = y_test[i:i+batch_size]
        Y = nn.forward(X)
        Y= Y.T.flatten()
        T = T.flatten()
        epoch_test_error += np.sum((Y-T)**2)
    # Compute loss
    train_error.append(epoch_train_error/train_samples)
    test_error.append(epoch_test_error/test_samples)
    
    # Backpropagation and weight updates
    # self.backward(X, y, y_pred)
    #plt.close()
    plt.clf()
    #fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')

    
    #out = nn.forward(patterns)
    #zz = out.reshape(gridsize, gridsize)

    # Plot the surface
    #ax.plot_wireframe(X_plot, Y_plot, Z_plot)
    #ax.plot_wireframe(X_plot, Y_plot, zz, color='red')
    #plt.pause(0.00000001)

    #ax.plot_wireframe(X, Y, zz)
    #if i % 100 == 0:
    print(f'Epoch {i}, Loss: {epoch_test_error}')
#plt.clf()
plt.plot(train_error)
plt.plot(test_error)
plt.show()
#train_dataset, test_dataset = create_dataset_20_80_from_classA(classA, classB)

plt.plot(x)

plt.show()
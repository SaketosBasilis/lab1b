import numpy as np
import matplotlib.pyplot as plt
import feed_forward_newral_network
from sklearn.model_selection import train_test_split

# Generate the x and y values (equivalent to MATLAB's [-5:0.5:5]')
x = np.arange(-5, 5.5, 0.5)  # note: the upper bound in np.arange() is exclusive
y = np.arange(-5, 5.5, 0.5)

# Create a meshgrid for x and y
X_plot, Y_plot = np.meshgrid(x, y)
# Calculate the z values
Z_plot = np.exp(-X_plot**2 * 0.1) * np.exp(-Y_plot**2 * 0.1) - 0.5
gridsize = len(x)
#patterns = np.vstack(X.flatten(),Y.flatten()).T
#targets = Z.reshape(-1,1).T
#print("patterns : ",patterns)
#print("targets : ",targets)
# Create the mesh plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_wireframe(X_plot, Y_plot, Z_plot)
plt.draw()

targets = Z_plot.reshape(1, -1).T  # equivalent to MATLAB's reshape(z, 1, ndata)
patterns = np.vstack([X_plot.ravel(), Y_plot.ravel()]).T

X_test, X_train,y_test,  y_train = train_test_split(patterns, targets, test_size=0.2, random_state=42)

train_error = []
test_error = []
print("patterns.shape : ",patterns.shape)
print("targets.shape : ",targets.shape)
epochs = 500
batch_size = 2
nn = feed_forward_newral_network.NeuralNetwork(input_size=2, hidden_size=25, output_size=1, hta_init=0.3,hta_final=0.0001,epochs=epochs)
train_samples = len(y_train)
test_samples = len(y_test)
for i in range(epochs):
    epoch_train_error = 0
    for i in range(0,train_samples,batch_size):
        X = X_train[i:i+batch_size,:]
        T = y_train[i:i+batch_size]
        Y = nn.forward(X)
        nn.backward(T)
        Y= Y.T.flatten()
        T = T.flatten()       
        
        epoch_train_error += np.sum((Y-T)**2)
    nn.update_sceduler()
    epoch_test_error = 0
    for i in range(0,test_samples,batch_size):
        X = X_test[i:i+batch_size,:]
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
    ax = fig.add_subplot(111, projection='3d')

    
    out = nn.forward(patterns)
    zz = out.reshape(gridsize, gridsize)

    # Plot the surface
    ax.set_xlabel('X')
    ax.set_xlim(-5, 5)
    ax.set_ylabel('Y')
    ax.set_ylim(-5, 5)
    ax.set_zlabel('Z')
    ax.plot_wireframe(X_plot, Y_plot, Z_plot,label="x")
    ax.plot_wireframe(X_plot, Y_plot, zz, color='red',label="aproximation")
    plt.pause(0.00000001)

    #ax.plot_wireframe(X, Y, zz)
    #if i % 100 == 0:
    print(f'Epoch {i}, Loss: {epoch_test_error}')
plt.clf()
plt.plot(train_error)
plt.plot(test_error)
plt.show()
#train_dataset, test_dataset = create_dataset_20_80_from_classA(classA, classB)

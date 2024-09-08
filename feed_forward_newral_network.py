import numpy as np
import matplotlib.pyplot as plt
import create_data
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, alpha = 0.4, hta=0.01):
        self.dw = np.zeros((input_size+ 1, hidden_size)) #TODO check the dimensions of matrix
        self.dv = np.zeros((hidden_size+1,output_size))

        self.w = np.random.normal(0,0.5,(input_size+ 1, hidden_size))
        self.v = np.random.normal(0,0.5,(hidden_size+1,output_size))
        self.hidden_size = hidden_size
        self.hta = hta
        self.alpha = alpha
    
    def add_bias(self, X):
        ones_column = np.ones((1,X.shape[0]))
        #print("ones_column.shape : ",ones_column.shape)
        #print("X : ",X.shape)
        X = np.hstack((X, ones_column.T))
        return X

    def phi(self, Χ):
        return 2/(1+np.exp(-Χ))-1
    def d_phi(self, phi):
        return (1+phi)*(1-phi)/2

    def forward(self, X):
        # Forward pass
        #print("X.shape : ",X.shape)
        X  = self.add_bias(X)
        #print("X.shape : ",X.shape)
        #print("self.w.shape : ",self.w.shape)
        self.hin = X @ self.w ;

        #print("self.hin.shape : ",self.hin.shape)
        #print("self.phi(self.hin).shape : ",self.phi(self.hin).shape)
        #print("self.add_bias(self.phi(self.hin)).shape : ",self.phi(self.hin).shape)
        self.hout =  self.add_bias(self.phi(self.hin))
        #print("self.hout.shape : ",self.hout.shape)
        #print("self.v : ",self.v.shape)
        self.oin = self.hout @ self.v
        #print("self.oin : ",self.oin.shape)
        self.out = self.phi(self.oin)
        self.X = X
        return self.out

    def backward(self, T):
        #print("self.out : ",self.out.shape)
        

        T = T.reshape(-1, 1)
        #print("T.shape : ",T)
        #print("self.out : ",self.out.shape)
        #print("self.d_phi(self.out).shape : ",self.d_phi(self.out).shape)
        #print("self.d_phi(self.out) : ",self.d_phi(self.out))
        #delta_o = n(self.out-T) self.d_phi(self.out)
        #print((self.out-T)*self.d_phi(self.out))
        #print("self.out.shape : ",self.out.shape)
        #print("(T).shape : ",(T).shape)
        #print("self.d_phi(self.out).shape : ",self.d_phi(self.out).shape)
        delta_o = (self.out-T)*self.d_phi(self.out)
        
        #print("self.v.T.shape : ",self.v.T.shape)
        #print("delta_o.shape : ",delta_o.shape)
        #print("self.hout.shape : ",self.hout.shape)
        #print("self.d_phi(self.hout) : ",self.d_phi(self.hout).shape)
        delta_h = (delta_o@self.v.T)*self.d_phi(self.hout)
        delta_h = delta_h[:,0:self.hidden_size]
        #print("delta_h.shape : ",delta_h.shape)
        #print("X.shape : ",X.shape)
        #print("delta_h.T@X.shape",(X.T@delta_h).shape)
        #print("w : ",self.w.shape)
        #print("delta_o.shape : ",delta_o.shape)
        self.dw = self.alpha*self.dw- (1-self.alpha)*(self.X.T@delta_h)
        self.dv = self.alpha*self.dv- (1-self.alpha)*(self.hout.T@delta_o)
       
        self.w = self.w + self.hta*self.dw
        self.v = self.v + self.hta*self.dv


    def train(self, epochs=1000, batch_size = 50):
        classA, classB = create_data.create_linsep_data(100)

        classA, classB = create_data.create_non_linsep_data(100)
        #train_dataset, test_dataset = create_dataset_20_80_from_classA(classA, classB)
        train_dataset, test_dataset = create_data.create_dataset(classA, classB) # ,classA_in_the_test,classB_in_the_test
        train_error = []
        test_error = []
        train_samples, dim = train_dataset.shape
        train_data = train_dataset[:,:2]
        train_labels = train_dataset[:,2]


        test_samples, dim = test_dataset.shape
        test_data = test_dataset[:,:2]
        test_labels = test_dataset[:,2]
        print(test_samples)
        for i in range(epochs):
            epoch_train_error = 0
            for i in range(0,train_samples,batch_size):
                X = train_data[i:i+batch_size,:]
                T = train_labels[i:i+batch_size]
                Y = self.forward(X)
                self.backward(T)
                Y= Y.T.flatten()
                print("Y : ",Y)
                print("T : ",T)
                
                
                epoch_train_error += np.sum((Y-T)**2)
            epoch_test_error = 0
            for i in range(0,test_samples,batch_size):
                X = test_data[i:i+batch_size,:]
                T = test_labels[i:i+batch_size]
                Y = self.forward(X)
                Y= Y.T.flatten()
                epoch_test_error += np.sum((Y-T)**2)
            # Compute loss
            train_error.append(epoch_train_error/train_samples)
            test_error.append(epoch_test_error/test_samples)
            
            # Backpropagation and weight updates
            # self.backward(X, y, y_pred)

            #if i % 100 == 0:
            print(f'Epoch {i}, Loss: {epoch_test_error}')
        #print("here")

        x_min, x_max = -4, 4
        y_min, y_max = -4, 4
        resolution = 0.01
        x_values = np.arange(x_min, x_max, resolution)
        y_values = np.arange(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x_values, y_values)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        print("grid_points.shape : ",grid_points.shape)
        predictions = self.forward(grid_points).reshape(xx.shape)
        print("predictions.shape : ",predictions.shape)
        plt.contour(xx, yy, predictions, levels=[0], colors='red')
        test_data = test_dataset[:,:2]
        test_labels = test_dataset[:,2]
        plt.contourf(xx, yy, predictions, cmap='coolwarm', alpha=0.6)  # Filled contour with colors
        plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap='coolwarm', edgecolor='k')  # Plot the data points
        plt.title('Neural Network Decision Boundary')
        plt.xlabel('Input Dimension 1')
        plt.ylabel('Input Dimension 2')
        plt.show()
        exit(1)
        plt.plot(train_error)
        plt.plot(test_error)
        plt.show()
# Example usage
if __name__ == "__main__":
    # Example dataset: XOR problem (just for illustration, you can replace it with your dataset)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
    y = np.array([0, 1, 1, 0])  # Target labels (binary classification)

    # One-hot encode the labels for softmax cross-entropy
    # y_encoded = np.eye(2)[y]

    # Initialize the network: 2 input features, 2 hidden units, 2 output units (binary classification)
    nn = NeuralNetwork(input_size=2, hidden_size=50, output_size=1, hta=0.01)

    # Train the model
    nn.train( epochs=200)
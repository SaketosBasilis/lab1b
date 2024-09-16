import numpy as np
import matplotlib.pyplot as plt
import create_data
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, alpha = 0.4, hta_init=0.1,hta_final=0.1,epochs =100):
        self.dw = np.zeros((input_size+ 1, hidden_size)) #TODO check the dimensions of matrix
        self.dv = np.zeros((hidden_size+1,output_size))
        self.w = np.random.normal(0,0.5,(input_size+ 1, hidden_size))
        self.v = np.random.normal(0,0.5,(hidden_size+1,output_size))
        self.hidden_size = hidden_size
        self.hta = hta_init
        self.hta_init = hta_init
        self.hta_final = hta_final
        self.epochs = epochs
        self.current_epoch = 0
        self.alpha = alpha
    def update_sceduler(self):
        self.current_epoch += 1
        self.hta = (self.hta_init-self.hta_final)*(self.epochs-self.current_epoch)/(self.epochs)+self.hta_final
        
    
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
        X  = self.add_bias(X)
        self.hin = X @ self.w ;
        self.hidden_layer_output = self.phi(self.hin)
        self.hout =  self.add_bias(self.hidden_layer_output)
        self.oin = self.hout @ self.v
        self.out = self.phi(self.oin)
        self.X = X
        return self.out

    def backward(self, T):
        delta_o = (self.out-T)*self.d_phi(self.out)

        delta_h = (delta_o@self.v.T)*self.d_phi(self.hout)
        delta_h = delta_h[:,0:self.hidden_size]

        self.dw = self.alpha*self.dw- (1-self.alpha)*(self.X.T@delta_h)
        self.dv = self.alpha*self.dv- (1-self.alpha)*(self.hout.T@delta_o)
       
        self.w = self.w + self.hta*self.dw
        self.v = self.v + self.hta*self.dv


    def train(self, epochs=1000, batch_size = 40):
        classA, classB = create_data.create_linsep_data(100)

        classA, classB = create_data.create_non_linsep_data(100)
        #train_dataset, test_dataset = create_data.create_dataset_20_80_from_classA(classA, classB)
        #print( train_dataset, test_dataset)
        train_dataset, test_dataset = create_data.create_dataset(classA, classB,0.25,0.25) # ,classA_in_the_test,classB_in_the_test
        train_error = []
        test_error = []
        train_samples, dim = train_dataset.shape
        train_data = train_dataset[:,:2]
        train_labels = train_dataset[:,2]


        test_samples, dim = test_dataset.shape
        test_data = test_dataset[:,:2]
        test_labels = test_dataset[:,2]

        for i in range(epochs):
            epoch_train_error = 0
            for i in range(0,train_samples,batch_size):
                X = train_data[i:i+batch_size,:]
                T = train_labels[i:i+batch_size]
                Y = self.forward(X)
                T = T.reshape(-1,1)
                self.backward(T)
                Y= Y.T.flatten()
                #print("Y : ",Y)
                #print("T : ",T)
                
                
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

            print(f'Epoch {i}, train Loss: {epoch_train_error}, test Loss: {epoch_test_error}')
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
        contour = plt.contourf(xx, yy, predictions, cmap='coolwarm', alpha=0.6)  # Filled contour with colors
        cbar = plt.colorbar(contour)
        cbar.set_label('NN values')  # Label for the color bar
        #    plt.scatter(test_data[test_labels == i, 0], test_data[test_labels == i, 1],color=test_labels[i], edgecolor='k', label=f'Class {i}')
        colors = ["blue","red"]
        labels= ["class A train","class B train"]
        for i in [-1,1]:
            plt.scatter(train_data[train_labels==i, 0], train_data[train_labels==i, 1], c=colors[1 if i==1 else 0], edgecolor='k',label=labels[1 if i==1 else 0], marker='o')  # Plot the data points
        colors = ["green","purple"]
        labels= ["class A test","class B test"]
        for i in [-1,1]:
            plt.scatter(test_data[test_labels==i, 0], test_data[test_labels==i, 1], c=colors[1 if i==1 else 0], edgecolor='k',label=labels[1 if i==1 else 0], marker='x')  # Plot the data points
        plt.legend()
        plt.title('Neural Network with hidden size : '+str(self.hidden_size))
        plt.savefig('./Neural Network with hidden size '+str(self.hidden_size)+".pdf",dpi=200)
        plt.xlabel('x')
        plt.ylabel('y')

        
        plt.show()
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
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, hta_init=0.01,hta_final=0.01)

    # Train the model
    nn.train( epochs=1000)
import numpy as np
import matplotlib.pyplot as plt


def create_linsep_data(n: int):
    mA = np.array([2., .5])
    mB = np.array([-2., 0.])
    sigmaA = 0.25
    sigmaB = 0.25
    classA = np.random.standard_normal((2, n)).T * sigmaA + mA
    classB = np.random.standard_normal((2, n)).T * sigmaB + mB
    return classA, classB
def create_gaussian_function_data():
    a = 5.5
    b = -0.
    x = np.arange(-5, 5.5, 0.5)  # note: the upper bound in np.arange() is exclusive
    y = np.arange(-5, 5.5, 0.5)

    # Create a meshgrid for x and y
    X, Y = np.meshgrid(x, y)

    # Calculate the z values
    Z = np.exp(-X**2 * 0.1) * np.exp(-Y**2 * 0.1) - 0.5
    dataset = np.vstack(X.flatten(),Y.flatten())
    dataset = np.vstack(dataset,Z.flatten())
    return dataset
def create_non_linsep_data(n: int, shuffle = True):

    mA = np.array([1., .3])
    mB = np.array([0., -0.1])
    sigmaA = 0.2
    sigmaB = 0.3
    classA = np.random.standard_normal((2, n)).T * sigmaA + mA
    classA[int(n/2):,0] =  classA[:int(n/2),0] -2* mA[0] 
    classB = np.random.standard_normal((2, n)).T * sigmaB + mB
    if shuffle:
        #print(classB.shape)
        indices = np.random.permutation(classB.shape[0])
        classA = classA[indices[:],:]
        #print(classB.shape)
        #print(indices)
        #exit(1)
    return classA, classB

def create_dataset_20_80_from_classA(classA, classB):
    #print("classA.shape : ",classA.shape)
    #print("classB.shape : ",classB.shape)
    classA = np.vstack((classA[:, 0], classA[:, 1]))
    classB = np.vstack((classB[:, 0], classB[:, 1]))
    classA_labels = np.full(classA.shape[1], -1)
    classB_labels = np.full(classB.shape[1], 1)
    classA = np.vstack((classA,classA_labels))
    classB = np.vstack((classB,classB_labels))
    #print("classA.shape : ",classA.shape)
    #print("classB.shape : ",classB.shape)
    subset1 = classA[:, classA[0, :] < 0]  # Subset where classA[1, :] < 0
    subset2 = classA[:, classA[0, :] > 0]  # Subset where classA[1, :] > 0

    # Determine the number of samples to select from each subset for the test set
    n_subset1 = int(0.2 * subset1.shape[1])  # 20% of total classA columns
    n_subset2 = int(0.8 * subset2.shape[1])  # 80% of total classA columns

    # Randomly sample from each subset for the test set
    test_set_subset1_indices = np.random.choice(subset1.shape[1], n_subset1, replace=False)
    test_set_subset2_indices = np.random.choice(subset2.shape[1], n_subset2, replace=False)
    
    test_set_subset1 = subset1[:, test_set_subset1_indices]
    test_set_subset2 = subset2[:, test_set_subset2_indices]

    # Combine the two sampled subsets to form the test set
    test_set = np.concatenate((test_set_subset1, test_set_subset2), axis=1)

    # Determine the remaining elements for the train set (rest of classA + classB)
    remaining_subset1 = np.delete(subset1, test_set_subset1_indices, axis=1)
    remaining_subset2 = np.delete(subset2, test_set_subset2_indices, axis=1)

    # Combine the remaining elements of classA and the entire classB to form the train set
    train_set_classA = np.concatenate((remaining_subset1, remaining_subset2), axis=1)
    train_set = np.concatenate((train_set_classA, classB), axis=1)

    # Shuffle the train set and the test set
    np.random.shuffle(train_set.T)
    np.random.shuffle(test_set.T)
    #print("train_set.shape : ",train_set.shape)
    #print("test_set.shape : ",test_set.shape)
    return train_set, test_set



def create_dataset(classA, classB, classA_percentage_on_the_test= 0.25, classΒ_percentage_on_the_test= 0.25 ):
    classA = np.vstack((classA[:, 0], classA[:, 1]))
    classA_labels = np.full(classA.shape[1], -1)
    classB = np.vstack((classB[:, 0], classB[:, 1]))
    classB_labels = np.full(classB.shape[1], 1)
    classA_data = np.vstack((classA,classA_labels))
    classB_data = np.vstack((classB,classB_labels))
    classA_percentage_on_the_train_set = (1 - classA_percentage_on_the_test)*classA.shape[1]
    #print("classA_percentage_on_the_train_set : ",classA_percentage_on_the_train_set)
    classB_percentage_on_the_train_set = (1 - classΒ_percentage_on_the_test)*classB.shape[1]
    #print("classB_percentage_on_the_train_set : ",classB_percentage_on_the_train_set)
    #print("classA_data : ",classA_data.shape )
    classA_train_data, classA_test_data = classA_data[:,:int(classA_percentage_on_the_train_set)],  classA_data[:,int(classA_percentage_on_the_train_set):]
    classB_train_data, classB_test_data = classB_data[:,:int(classB_percentage_on_the_train_set)],  classB_data[:,int(classB_percentage_on_the_train_set):]

    train_dataset = np.hstack((classA_train_data, classB_train_data))
    indices = np.random.permutation(train_dataset.shape[1])
    train_dataset = train_dataset[:,indices[:]]


    test_dataset = np.hstack((classA_test_data, classB_test_data))
    indices = np.random.permutation(test_dataset.shape[1])
    test_dataset = test_dataset[:,indices[:]]

    #scatter = plt.scatter(train_dataset[0,:],train_dataset[1,:], c=train_dataset[2,:])
    #scatter = plt.scatter(test_dataset[0,:],test_dataset[1,:], c=test_dataset[2,:])

    # Add color bar to show label colors

    return train_dataset.T, test_dataset.T

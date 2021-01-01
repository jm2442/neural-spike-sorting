# Import libraries required
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Import modules for functions required
from modules import plot

def NeuralNet(train_data, train_label, test_data, test_label, num_layers, num_neurons, act_function, alpha, learn_rate_type, learn_rate_init, max_iter, plot_on=False, evaluate=True):
    # Returns either the predictions from a trained multi-layer perceptron model or the trained model itself
    
    # Because of the nature of the optimiser, build the structure of the neural network by adding the number of neurons the correct number of layers
    hls=[]
    for x in range(int(num_layers)):
        hls.append(int(num_neurons))
    hls = tuple(hls)
    
    # Extract data for fitting the model
    train_X = [x[0] for x in train_data]
    train_Y = train_label[:]

    # Build the model with the chosen parameters
    MLP = MLPClassifier(hidden_layer_sizes=hls, activation=act_function, alpha=alpha, learning_rate=learn_rate_type, learning_rate_init=learn_rate_init, max_iter=max_iter)

    # Fit the model with the training data
    MLP.fit(train_X, train_Y)

    # If evaluating, return the model's predictions for the test data
    if evaluate:
        test_X = [x[0] for x in test_data]

        if plot_on: 
            # Plot the confusion matrix for the classifier's performance
            plot.confusion_matrix(MLP, test_X, test_label)

        # Predict the output for the test data and return it alongside its corresponding index location
        pred_Y = MLP.predict(test_X)
        return [[pred_Y[x], test_data[x][1]] for x in range(len(pred_Y))]
    else:
        # Return the model itself
        return MLP

def KNearNeighbor(train_data, train_label, test_data, test_label, neighbors, weights, plot_on=False, evaluate=True):
    # Returns either the predictions from a trained k nearest neighbours model or the trained model itself

    # Extract data for fitting the model
    train_X = [x[0] for x in train_data]
    train_Y = train_label[:]

    # Build the model with the chosen parameters
    KNN = KNeighborsClassifier(n_neighbors=int(neighbors), weights=weights)

    # Fit the model with the training data
    KNN.fit(train_X, train_Y)

    # If evaluating, return the model's predictions for the test data
    if evaluate:
        test_X = [x[0] for x in test_data]

        if plot_on:
            # Plot the confusion matrix for the classifier's performance
            plot.confusion_matrix(KNN, test_X, test_label)

        # Predict the output for the test data and return it alongside its corresponding index location
        pred_Y = KNN.predict(test_X)
        return [[pred_Y[x], test_data[x][1]] for x in range(len(pred_Y))]
    else:
        # Return the model itself
        return KNN
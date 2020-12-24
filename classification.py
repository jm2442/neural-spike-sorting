from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import plot
import numpy as np


def NeuralNet(train_data, train_label, test_data, test_label, num_layers, num_neurons, act_function, alpha, learn_rate_type, plot_on=False, evaluate=True):
    
    hls=[]
    for x in range(int(num_layers)):
        hls.append(int(num_neurons))
    hls = tuple(hls)
    
    act_funcs = ['identity', 'logistic', 'tanh', 'relu']
    act = act_funcs[int(act_function)]

    learn_rate = ['constant', 'invscaling', 'adaptive']
    learn = learn_rate[int(learn_rate_type)]

    train_X = [x[0] for x in train_data]
    train_Y = train_label[:]

    # googoo = train_data[:,0]

    MLP = MLPClassifier(hidden_layer_sizes=hls, activation=act, alpha=alpha, learning_rate=learn)
    MLP.fit(train_X, train_Y)
    if evaluate:

        test_X = [x[0] for x in test_data]
        if plot_on: 
            plot.confusion_matrix(MLP, test_X, test_label)

        pred_Y = MLP.predict(test_X)

        return [[pred_Y[x], test_data[x][1]] for x in range(len(pred_Y))]
    else:
        return MLP

def KNearNeighbor(train_data, train_label, test_data, test_label, neighbors=20, plot_on=False, evaluate=True):

    train_X = [x[0] for x in train_data]
    train_Y = train_label[:]

    KNN = KNeighborsClassifier(n_neighbors=int(neighbors))
    KNN.fit(train_X, train_Y)

    if evaluate:
        test_X = [x[0] for x in test_data]
        if plot_on: 
            plot.confusion_matrix(KNN, test_X, test_label)

        pred_Y = KNN.predict(test_X)

        return [[pred_Y[x], test_data[x][1]] for x in range(len(pred_Y))]
    else:
        return KNN
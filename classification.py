from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


def NeuralNet(train_data, train_label, test_data, num_layers, num_neurons, act_function, alpha, learn_rate_type):
    
    hls=[]
    for x in range(int(num_layers)):
        hls.append(int(num_neurons))
    hls = tuple(hls)
    
    act_funcs = ['identity', 'logistic', 'tanh', 'relu']
    act = act_funcs[int(act_function)]

    learn_rate = ['constant', 'invscaling', 'adaptive']
    learn = learn_rate[int(learn_rate_type)]

    MLP = MLPClassifier(hidden_layer_sizes=hls, activation=act, alpha=alpha, learning_rate=learn)
    MLP.fit(train_data, train_label)
    return MLP.predict(test_data)

def KNearNeighbor(train_data, train_label, test_data, neighbors=20):
    KNN = KNeighborsClassifier(n_neighbors=int(neighbors))
    KNN.fit(train_data, train_label)
    return KNN.predict(test_data)
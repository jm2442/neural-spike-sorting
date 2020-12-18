from sklearn.neighbors import KNeighborsClassifier

def KNN(train_data, train_label, test_data, neighbors=20):
    KNN = KNeighborsClassifier(n_neighbors=neighbors)
    KNN.fit(train_data, train_label)
    return KNN.predict(test_data)
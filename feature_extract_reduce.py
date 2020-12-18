import sklearn as sk
from sklearn.decomposition import PCA

def dimension_reducer(data_samples, dimensions=3):

    # Apply min-max scaling
    scaler = sk.preprocessing.MinMaxScaler()
    scaled_data_samples = scaler.fit_transform(data_samples)

    # Do PCA
    pc_analysis = PCA(n_components=dimensions)
    pca = pc_analysis.fit_transform(scaled_data_samples)

    return pca
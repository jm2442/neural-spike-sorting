# Import libraries required
import sklearn as sk
from sklearn.decomposition import PCA

def dimension_reducer(data_samples, dimensions=3):
    # Returns the output of the principal component analysis to give a smaller dimmension size

    # Apply min-max scaling
    scaler = sk.preprocessing.MinMaxScaler()
    scaled_data_samples = scaler.fit_transform(data_samples)

    # Perform Principal Component Analysis
    pc_analysis = PCA(n_components=dimensions)
    pca = pc_analysis.fit_transform(scaled_data_samples)

    return pca
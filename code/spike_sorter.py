# Import libraries required
from sklearn.model_selection import train_test_split, KFold
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import math
import pickle
# Import code for functions required
from code import filtering as filt
from code import spike_detection as spdt
from code import alignment as align
from code import feature_extract_reduce as feat_ex_reduce
from code import classification as classifier
from code import plot
from code import performance_metrics as metrics
    
def spike_sorter(params, fixed_arguments, clf_type, print_on, plot_on, evaluate=True, x_start = 0.24, x_end = 0.29):
    # Returns an evaluate score on a trained model's performance or the trained model itself

    # Print parameters which have been input
    print("-"*20)
    print(params)

    # Extract parameters for sorter depending on classification method chosen
    if clf_type == 2:
        samp_freq, window_size, act_function, alpha, learn_rate_type, learn_rate_init, max_iter = fixed_arguments
        low_cutoff, high_cutoff, smooth_size, edo_thresh_factor,num_layers, num_neurons = params
    elif clf_type == 3:
        samp_freq, window_size, pca_dim, weights = fixed_arguments
        low_cutoff, high_cutoff, smooth_size, edo_thresh_factor,num_neighbors = params

    # Load corresponding dataset from .mat file provided
    mat = spio.loadmat('../neural-spike-sorting/datasets/training.mat', squeeze_me=True)

    # Assign loaded data to variables
    d = mat['d']
    idx_test, class_test = (list(t) for t in zip(*sorted(zip(mat['Index'], mat['Class']))))

    # Apply a bandpass and a Savitzky-golay filter to remove worst of the noise
    time, filt_d, smth_d = filt.signal_processing(d, low_cutoff, high_cutoff, smooth_size, samp_freq)

    # Determine the location of the peaks' idxes
    peak_idxes, edo_d, edo_thresh, smth_thresh = spdt.peak_detector(smth_d, edo_thresh_factor)
    idx_train = list(peak_idxes[0])

    # Compare against provided indexes to determine the accuracy of the peak detection
    all_peaks, peak_loc_success = metrics.peak_location_accuracy(idx_test, idx_train, class_test, print_on)

    # Find the labels and indexes of the peaks that have been correctly identified
    found_pk_lbl = [x[1] for x in all_peaks if x[2]]
    found_pk_idx = [x[0] for x in all_peaks if x[2]]

    # Extract each spike as a sample window
    d_samp = align.spike_extractor(smth_d, found_pk_idx, window_size)

    # Evaluate will be true for all times except when a model is to be trained on the training dataset to be run on the submission dataset
    if evaluate:

        # Set number of k fold splits to account for the proportion of training to test data, 5 = 80/20, 4 = 75/25 etc.
        k_splits = 5
        if clf_type == 2:
            
            # Split the data by index of k iterations to preform k fold cross validation
            kf  = KFold(n_splits=k_splits)
            kth_score = []
            for train_k, test_k in kf.split(d_samp):

                # Split training and test data
                train_d, test_d = np.array(d_samp)[train_k], np.array(d_samp)[test_k]
                train_lbl, test_lbl = np.array(found_pk_lbl)[train_k], np.array(found_pk_lbl)[test_k]

                # Perform classification using a Multi-Layer Perceptron
                pred_lbl = classifier.NeuralNet(train_d, train_lbl, test_d, test_lbl, num_layers, num_neurons, act_function, alpha, learn_rate_type, learn_rate_init, max_iter, plot_on)

                # Compute the metrics of the classifcation and add to list of k number of scores
                f1_score = metrics.peak_classification(test_lbl, pred_lbl, print_on)

                kth_score.append(f1_score)

        elif clf_type == 3:

            # Preform PCA to extract the most important features and reduce dimension
            d_samp_window = [x[0] for x in d_samp]
            pca_out = feat_ex_reduce.dimension_reducer(d_samp_window, pca_dim)
            pca = [[pca_out[x], d_samp[x][1]] for x in range(len(pca_out))]

            # Split the data by index of k iterations to preform k fold cross validation
            kf  = KFold(n_splits=k_splits)
            kth_score = []
            for train_k, test_k in kf.split(d_samp):

                # Split training and test data
                train_d, test_d = np.array(pca)[train_k], np.array(pca)[test_k]
                train_lbl, test_lbl = np.array(found_pk_lbl)[train_k], np.array(found_pk_lbl)[test_k]

                # Perform classification using K Nearest Neighbours
                pred_lbl = classifier.KNearNeighbor(train_d, train_lbl, test_d, test_lbl, num_neighbors, weights, plot_on)

                # Compute the metrics of the classifcation and add to list of k number of scores
                f1_score = metrics.peak_classification(test_lbl, pred_lbl, print_on)

                kth_score.append(f1_score)

        # Avg the k number of scores using the mean of their values
        mean_f1_score = np.mean(kth_score)
        std_f1_score = np.std(kth_score)
        if print_on:
            print("*"*20)
            print("Mean Weighted F1 score (%) = "+ str(round(mean_f1_score*100, 2)))
            print("Model Bias = "+ str(round((1-mean_f1_score), 4)))
            print("Model Variance = "+ str(round(std_f1_score, 4)))

        ##### PLOTTING 
        if plot_on:
            
            # Extracting data for plots
            no_lbl_test_data = [x[0] for x in test_d]
            no_idx_pred_lbl = [x[0] for x in pred_lbl]
            d_samp_window = [x[0] for x in d_samp]
            time_test = [time[int(peak)] for peak in idx_test]
            train_test_samples = [[d_samp_window[x], found_pk_lbl[x]] for x in range(len(d_samp_window))]

            # Plot filtering and peak detection over set interval
            plot.filter_and_detection(x_start, x_end, time, d, time_test, idx_train, idx_test, filt_d, smth_d, smth_thresh, edo_d, edo_thresh)

            # Plot extracted labelled spikes from training data
            plot.samples(train_test_samples, 1)

            if clf_type == 2:
                # Plot predictions from MLP classifier
                plot.MLP(no_lbl_test_data, no_idx_pred_lbl, d_samp_window)
            elif clf_type == 3:
                # Plot predictions from PCA & KNN classifier
                plot.PCA(pca_out)
                plot.KNN(no_lbl_test_data, no_idx_pred_lbl, np.array(d_samp_window)[test_k])

            plt.show()
        
        print("*"*20)
        print("Total System Score = " + str(round(peak_loc_success * mean_f1_score * 100, 2)))
        print("*"*20)

        return peak_loc_success * mean_f1_score

    else:

        if clf_type == 2:

            train_d = np.array(d_samp)
            train_lbl = np.array(found_pk_lbl)
            test_d = []
            test_lbl = []

            # Build, train and return MLP model to be applied to submission dataset
            MLP = classifier.NeuralNet(train_d, train_lbl, test_d, test_lbl, num_layers, num_neurons, act_function, alpha, learn_rate_type, learn_rate_init, max_iter, plot_on, evaluate=False)

            return MLP

        elif clf_type == 3:
            # Preform PCA to extract the most important features and reduce dimension
            d_samp_window = [x[0] for x in d_samp]
            pca = feat_ex_reduce.dimension_reducer(d_samp_window, pca_dim)
            pca = [[pca[x], d_samp[x][1]] for x in range(len(pca))]

            train_d = np.array(pca)
            train_lbl = np.array(found_pk_lbl)
            test_d = []
            test_lbl = []

            # Build, train and return KNN model to be applied to submission dataset
            KNN = classifier.KNearNeighbor(train_d, train_lbl, test_d, test_lbl, num_neighbors, weights, plot_on, evaluate=False)

            return KNN
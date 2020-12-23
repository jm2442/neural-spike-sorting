import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import math
from sklearn.model_selection import train_test_split, KFold
###
import filtering as filt
import spike_detection as spdt
import alignment as align
import feature_extract_reduce as feat_ex_reduce
import classification as classifier
import plot
import performance_metrics as metrics
    
def spike_sorter(params):#, args):

    part = 3
    print_on = True
    plot_on = True
    print("-"*20)
    print(params)

    # Extract parameters for sorter depending on classification method chosen
    if part == 2:
        low_cutoff, high_cutoff, smooth_size, edo_thresh_factor, window_size,  num_layers, num_neurons, act_function, alpha, learn_rate_type = params
    elif part == 3:
        low_cutoff, high_cutoff, smooth_size, edo_thresh_factor, window_size,num_neighbors = params#[0]

    # Load corresponding dataset from .mat file provided
    mat = spio.loadmat('../neural-spike-sorting/datasets/training.mat', squeeze_me=True)

    # Assign loaded data to variables
    d = mat['d']
    idx_test, class_test = (list(t) for t in zip(*sorted(zip(mat['Index'], mat['Class']))))

    # Set the sampling frequency that is given to us, convert frequency sample into cumulative time
    samp_freq = 25000
    time = list(np.linspace(0, len(d)*1/samp_freq, len(d)))
    time_test = [time[int(peak)] for peak in idx_test]
    # individual_sample_time = 2.5e-3

    # Apply a bandpass and a Savitzky-golay filter to remove worst of the noise
    filt_d  = filt.bandpass(d, samp_freq, low_cutoff, high_cutoff)
    smth_d = filt.smoothing(filt_d, smooth_size)

    # Determine the location of the peaks' idxes
    peak_idxes, edo_d, edo_thresh, smth_thresh = spdt.peak_detector(smth_d, edo_thresh_factor)
    idx_train = list(peak_idxes[0])

    # Compare against provided indexes to determine the accuracy of the peak detection
    all_peaks, peak_loc_success = spdt.peak_location_accuracy(idx_test, idx_train, class_test, print_on)

    # Find the labels and indexes of the peaks that have been correctly identified
    found_pk_lbl = [x[1] for x in all_peaks if x[2]]
    found_pk_idx = [x[0] for x in all_peaks if x[2]]

    # Extract each spike as a sample window
    spike_samp_arr = align.spike_extractor(smth_d, found_pk_idx, window_size)
    d_samp = spike_samp_arr#[x[0] for x in spike_samp_arr]

    # Set number of k fold splits to account for the proportion of training to test data, 5 = 80/20, 4 = 75/25 etc.
    k_splits = 5

    if part == 2:
        
        # Split the data by index of k iterations to preform k fold cross validation
        kf  = KFold(n_splits=k_splits)
        kth_score = []
        for train_k, test_k in kf.split(d_samp):

            # Split training and test data
            train_d, test_d = np.array(d_samp)[train_k], np.array(d_samp)[test_k]
            train_lbl, test_lbl = np.array(found_pk_lbl)[train_k], np.array(found_pk_lbl)[test_k]


        
            # Perform classification using a Multi-Layer Perceptron
            pred_lbl = classifier.NeuralNet(train_d, train_lbl, test_d, test_lbl, num_layers, num_neurons, act_function, alpha, learn_rate_type, plot_on)

            # Compute the metrics of the classifcation and add to list of k number of scores
            f1_score, spike_metrics = metrics.peak_classification(test_lbl, pred_lbl, print_on)

            # if plot_on:
            #     no_lbl_test_data = [x[0] for x in test_d]
            #     no_idx_pred_lbl = [x[0] for x in pred_lbl]
            #     d_samp_window = [x[0] for x in d_samp]
            #     plot.MLP(no_lbl_test_data, no_idx_pred_lbl, d_samp_window)

            kth_score.append(f1_score)

    elif part == 3:

        # Preform PCA to extract the most important features and reduce dimension
        pca_dim = 20
        d_samp_window = [x[0] for x in d_samp]
        pca = feat_ex_reduce.dimension_reducer(d_samp_window, pca_dim)
        pca = [[pca[x], d_samp[x][1]] for x in range(len(pca))]

        # Split the data by index of k iterations to preform k fold cross validation
        kf  = KFold(n_splits=k_splits)
        kth_score = []
        for train_k, test_k in kf.split(d_samp):

            # Split training and test data
            train_d, test_d = np.array(pca)[train_k], np.array(pca)[test_k]
            train_lbl, test_lbl = np.array(found_pk_lbl)[train_k], np.array(found_pk_lbl)[test_k]

            # Perform classification using K Nearest Neighbours
            pred_lbl = classifier.KNearNeighbor(train_d, train_lbl, test_d, test_lbl, num_neighbors, plot_on)

            # Compute the metrics of the classifcation and add to list of k number of scores
            f1_score, spike_metrics = metrics.peak_classification(test_lbl, pred_lbl, print_on)

            # if plot_on:
            #     no_lbl_test_data = [x[0] for x in test_d]
            #     no_idx_pred_lbl = [x[0] for x in pred_lbl]
            #     plot.KNN(no_lbl_test_data, no_idx_pred_lbl, d_samp_window)
            kth_score.append(f1_score)

    # Avg the k number of scores using the mean of their values
    avg_f1_score = sum(kth_score)/k_splits
    if print_on:
        print("*"*20)
        print("Average Weighted F1 score (%) = "+ str(round(avg_f1_score*100, 2)))
        # print("*"*20)

    ### PLOTTING 
    if plot_on:
        x_start = 10
        x_end = 12
        plot.filter_and_detection(x_start, x_end, time, d, time_test, idx_train, idx_test, filt_d, smth_d, smth_thresh, edo_d, edo_thresh)
        # plot.samples(d_samp, 100)

        no_lbl_test_data = [x[0] for x in test_d]
        no_idx_pred_lbl = [x[0] for x in pred_lbl]
        d_samp_window = [x[0] for x in d_samp]
        if part == 2:
            # plot.PCA(pca)
            plot.MLP(no_lbl_test_data, no_idx_pred_lbl, d_samp_window)
        elif part == 3:
            
            plot.KNN(no_lbl_test_data, no_idx_pred_lbl, np.array(d_samp_window)[test_k])

        plt.show()
    
    print("*"*20)
    print("Average Score = " + str(round(peak_loc_success * avg_f1_score * 100, 2)))
    print("*"*20)

    return peak_loc_success * avg_f1_score

# TO DO 
# PLOT LABELS< LEGENDS< ETC
# Offset from peak to rising edge
# 50 threshold????
# Bias and threshold of model
# CONFUSION LOCATION
# Configure code to run submission dataset
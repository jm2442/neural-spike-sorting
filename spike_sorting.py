import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import math
from sklearn.model_selection import train_test_split
###
import filtering as filt
import spike_detection as spdt
import alignment as align
import feature_extract_reduce as feat_ex_reduce
import classification as classifier
import plot
import performance_metrics as metrics
    
def spike_sorter(params):#, args):

    part = 2
    print_on = False
    print(params)

    low_cutoff = 46
    high_cutoff = 3467
    smooth_size = 17
    edo_thresh_factor = 13
    window_size = 22

    if part == 2:

        num_layers, num_neurons, act_function, alpha, learn_rate_type = params

    elif part == 3:

        num_neighbors = params

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

    # Find the labels of the peaks that have been correctly identified
    found_pk_lbl = [x[1] for x in all_peaks if x[2]]

    # Extract each spike as a sample window
    spike_samp_arr = align.spike_extractor(smth_d, idx_train, window_size)
    d_samp = [x[0] for x in spike_samp_arr]


    test_percent = 0.25

    if part == 2:

        train_d, test_d, train_lbl, test_lbl = train_test_split(d_samp, found_pk_lbl, test_size=test_percent)
        pred_lbl = classifier.NeuralNet(train_d, train_lbl, test_d, num_layers, num_neurons, act_function, alpha, learn_rate_type)

    elif part == 3:
        # Preform PCA to extract the most important features and reduce dimension
        pca_dim = 3
        pca = feat_ex_reduce.dimension_reducer(d_samp, pca_dim)

        # Split training and test
        train_d, test_d, train_lbl, test_lbl = train_test_split(pca[:,0:2], found_pk_lbl, test_size=test_percent)

        # Preform K Nearest Neighbours classification
        pred_lbl = classifier.KNearNeighbor(train_d, train_lbl, test_d, num_neighbors)

    score = metrics.peak_classification(test_lbl, pred_lbl, print_on)

    ### PLOTTING 
    plot_on = False
    if plot_on:
        x_start = 0
        x_end = 2
        plot.filter_and_detection(x_start, x_end, time, d, time_test, idx_train, idx_test, filt_d, smth_d, smth_thresh, edo_d, edo_thresh)
        plot.samples(d_samp, 100)
        # plot.PCA(pca)
        plot.KNN(test_d, pred_lbl, d_samp)
        plt.show()
    
    print("Final Score = " + str(round(peak_loc_success * score * 100, 2)))

    return peak_loc_success #* score

# TO DO 
# PLOT LABELS< LEGENDS< ETC
# Offset from peak to rising edge
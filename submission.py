# Import libraries required
import pickle
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
# Import modules for functions required
import training as opt
from modules import spike_sorter as spsrt
from modules import filtering as filt
from modules import spike_detection as spdt
from modules import alignment as align
from modules import feature_extract_reduce as feat_ex_reduce
from modules import plot

# Set the Classifier of choice. 2 (MLP) or 3 (KNN)
clf_type = 3

# Set if to plot the output graphs or not
plot_on = True

# Obtain parameters for either classifier from where optimisation has been performed
params = opt.parameters(clf_type)
if clf_type == 2:
    low_cutoff, high_cutoff, smooth_size, edo_thresh_factor, window_size,  num_layers, num_neurons, act_function, alpha, learn_rate_type = params
elif clf_type == 3:
    low_cutoff, high_cutoff, smooth_size, edo_thresh_factor, window_size,num_neighbors = params

# Load corresponding dataset from .mat file provided
mat = spio.loadmat('../neural-spike-sorting/datasets/submission.mat', squeeze_me=True)

# Assign loaded data to variables
d = mat['d']

# Filter noisey data and smooth to help classifier distinguish shapes
time, filt_d, smth_d = filt.signal_processing(d, low_cutoff, high_cutoff, smooth_size)

# Determine the location of the peaks' idxes
peak_idxes, edo_d, edo_thresh, smth_thresh = spdt.peak_detector(smth_d, edo_thresh_factor)

# Extract each spike as a sample window
found_pk_lbl = [x for x in peak_idxes[0]]
d_samp = np.array(align.spike_extractor(smth_d, found_pk_lbl, window_size))

# Try to see if a pretrained saved model exists to save computation
try:
    if clf_type == 2:
        filename = '../neural-spike-sorting/models/MLP.pkl'
    elif clf_type == 3:
        filename = '../neural-spike-sorting/models/KNN.pkl'     
    with open(filename, 'rb') as f:
        trained_clf = pickle.load(f)
    print("Saved model found, classifying submission dataset")
except Exception as ex:
    # If no saved model is found perform retraining with the optimised parameters for the training dataset
    print("No saved trained model found, retraining of training dataset")
    trained_clf = spsrt.spike_sorter(params, clf_type, print_on=False, plot_on=False, evaluate=False)

# Extract just the window data and not the peak index
d_samp_window = [x[0] for x in d_samp]

# If KNN has been chosen then perform PCA
if clf_type == 2:
    input_samples = d_samp_window[:]
    filestring = 'MLP'
elif clf_type == 3:
    pca_dim = trained_clf.n_features_in_
    input_samples = feat_ex_reduce.dimension_reducer(d_samp_window, pca_dim)
    filestring = 'KNN'

print('-'*10)
print("Model ready, running submission dataset")

# Using the model trained on the training dataset, predict the labels of the spikes
pred_lbl = trained_clf.predict(input_samples)

# Integrate the corresponding spike indexes to the spike class
predictions = [[pred_lbl[x], d_samp[x][1]] for x in range(len(pred_lbl))]
output = {
    "Class":[x[0] for x in predictions],
    "Index":[x[1] for x in predictions]
}

# Output the dataset for the chosen classifier
spio.savemat('../neural-spike-sorting/datasets/output_'+filestring+'.mat', output)

if plot_on:

    # Show the signal over set time interval
    xstart = 0
    xend = 0.8

    # Plot the output of the filtering and peak detection performed over the set interval
    idx_train = list(peak_idxes[0])
    plot.filter_and_detection(xstart, xend, time, d, time_test=[], index_train=idx_train, index_test=[], filtered_data=filt_d, smoothed_data=smth_d, smoothed_threshold=smth_thresh, edo_data=edo_d, edo_threshold=edo_thresh, training=False)

    # Plot the output spike train
    num_spike = plot.spike_train(xstart, xend, time, d, predictions)

    # Print number of peaks
    print("Number of 1st neuron: " + str(num_spike[0]))
    print("Number of 2nd neuron: " + str(num_spike[1]))
    print("Number of 3rd neuron: " + str(num_spike[2]))
    print("Number of 4th neuron: " + str(num_spike[3]))

    print("Total number of peaks found: " + str(len(predictions)))

    # Plot the average waveform for each of the classified neurons
    if clf_type == 2:
        plot.MLP(input_samples, pred_lbl, d_samp_window)
    elif clf_type == 3:
        plot.KNN(input_samples, pred_lbl, d_samp_window)

    # Show all drawn plots at the end of the script
    plt.show()

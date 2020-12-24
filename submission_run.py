import pickle
import scipy.io as spio
import numpy as np
#
import optimisation as opt
import spike_sorting as spsrt
import filtering as filt
import spike_detection as spdt
import alignment as align
import feature_extract_reduce as feat_ex_reduce

clf_type = 3
params = opt.parameters(clf_type)
if clf_type == 2:
    low_cutoff, high_cutoff, smooth_size, edo_thresh_factor, window_size,  num_layers, num_neurons, act_function, alpha, learn_rate_type = params
elif clf_type == 3:
    low_cutoff, high_cutoff, smooth_size, edo_thresh_factor, window_size,num_neighbors = params

# Load corresponding dataset from .mat file provided
mat = spio.loadmat('../neural-spike-sorting/datasets/submission.mat', squeeze_me=True)

# Assign loaded data to variables
d = mat['d']

# filtering
time, filt_d, smth_d = filt.signal_processing(d, low_cutoff, high_cutoff, smooth_size)

# Determine the location of the peaks' idxes
peak_idxes, edo_d, edo_thresh, smth_thresh = spdt.peak_detector(smth_d, edo_thresh_factor)

# Extract each spike as a sample window
found_pk_lbl = [x for x in peak_idxes[0]]
d_samp = np.array(align.spike_extractor(smth_d, found_pk_lbl, window_size))

try:
    if clf_type == 2:
        filename = '../neural-spike-sorting/models/MLP.pkl'
    elif clf_type == 3:
        filename = '../neural-spike-sorting/models/KNN.pkl'     
    with open(filename, 'rb') as f:
        trained_clf = pickle.load(f)
    print("Saved model found, beginning classification of submission dataset")
    
except Exception as ex:
    print("No saved trained model found, retraining of training dataset")
    trained_clf = spsrt.spike_sorter(params, clf_type, print_on=False, plot_on=False, evaluate=False)

if clf_type == 2:
    input_samples = [x[0] for x in d_samp]

elif clf_type == 3:
    pca_dim = 3
    d_samp_window = [x[0] for x in d_samp]
    input_samples = feat_ex_reduce.dimension_reducer(d_samp_window, pca_dim)
    # pca = [[pca[x], d_samp[x][1]] for x in range(len(pca))]


# huh =input_samples[1:len(input_samples)-1]
pred_lbl = trained_clf.predict(input_samples)
predictions = [[pred_lbl[x], d_samp[x][1]] for x in range(len(pred_lbl))]
print("WASSUP")

# Run KNN model trained by training data

# output file

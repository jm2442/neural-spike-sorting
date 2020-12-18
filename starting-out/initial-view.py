import scipy.io as spio
# import scipy.stats as stat
from scipy.signal import butter, lfilter, find_peaks, savgol_filter
import matplotlib.pyplot as plt 
import numpy as np
import sklearn as sk
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def bandpass_filter(data, samp_freq, low=1, high=2000, order=2):
    nyq_freq = samp_freq/2

    low_band = low/nyq_freq
    high_band = high/nyq_freq

    b, a  = butter(order, [low_band, high_band], btype='band')

    filtered_data = lfilter(b, a, data)

    return filtered_data

def threshold_finder(filtered_data, thresh_factor=5):
    scaled_abs_signal = [abs(x)/0.6745 for x in filtered_data]
    sigma_n = np.median(scaled_abs_signal)
    thr = thresh_factor * sigma_n
    return thr

def hilbert_transform(x):
    """
    compute the discrete hilbert transform, as defined in [1].

    based on code from https://github.com/otoolej/envelope_derivative_operator/blob/master/energy_operators/edo.py (Accessed 16/12/20)

    [1] JM O' Toole, A Temko, NJ Stevenson, “Assessing instantaneous energy in the EEG: a non-negative, frequency-weighted energy operator”, IEEE Int. Conf.  on Eng. in Medicine and Biology, Chicago, August 2014
    """

    xlen = len(x)
    xmid = np.ceil(xlen / 2)
    k = np.arange(xlen)

    # Build the Hilbert transform in the frequency domain:
    H = -1j * np.sign(xmid - k) * np.sign(k)
    x_hilb = np.fft.ifft(np.multiply(np.fft.fft(x), H))
    x_hilb = np.real(x_hilb)

    return(x_hilb)

def envel_deriv_operator(x):
    """
    compute the envelope derivative operator (EDO), as defined in [1].

    based on code from https://github.com/otoolej/envelope_derivative_operator/blob/master/energy_operators/edo.py (Accessed 16/12/20)

    [1] JM O' Toole, A Temko, NJ Stevenson, “Assessing instantaneous energy in the EEG: a non-negative, frequency-weighted energy operator”, IEEE Int. Conf.  on Eng. in Medicine and Biology, Chicago, August 2014
    """

    # Make sure x is an even length
    initial_xlen = len(x)
    if (initial_xlen % 2) != 0:
        x = np.hstack((x,0))
    
    xlen = len(x)
    nl = np.arange(1, xlen-1)
    xx = np.zeros(xlen)

    # Calculate the Hilbert Transform
    h = hilbert_transform(x)

    # Implement with the central finite difference equation
    xx[nl] = (((x[nl+1] ** 2) + (x[nl-1] ** 2) + (h[nl+1] ** 2) + (h[nl-1] ** 2)) / 4) + ((x[nl+1] * x[nl-1] + h[nl+1] * h[nl-1]) / 2)

    # Trim and zero pads at the ends
    x_edo = np.pad(xx[2:(len(xx) - 2)], (2, 2), 'constant', constant_values=(0, 0))

    return x_edo[0:initial_xlen]
    
def spike_detector(filtered_data):

    energy_x = envel_deriv_operator(filtered_data)

    thresh_factor = 10
    threshold = threshold_finder(energy_x, thresh_factor)
    thresh_factor = 5
    threshold2 = threshold_finder(filtered_data,thresh_factor)

    peak_indices = find_peaks(energy_x, threshold)
    return peak_indices, energy_x, threshold, threshold2

def smoothing_filter(filtered_data, window=21):

    smoothed_data = savgol_filter(filtered_data, window, 3)

    return smoothed_data
    
def spike_extractor(filtered_data, peaks, window_size=64):

    # Do initial extraction of spike according to peak from energy operator
    single_sample_array = []
    # multiple_sample_array = []
    window_midpoint = window_size//2

    for x in range(len(peaks)):

        current_peak = peaks[x]

        window = filtered_data[current_peak-5:current_peak+5]

        max_point  = np.where(filtered_data==max(window))
        aligned_max = max_point[0][0]

        aligned_window = filtered_data[aligned_max-window_midpoint:aligned_max+window_midpoint]

        if x+1 < len(peaks):
            if peaks[x-1] < current_peak-(window_midpoint*2) and peaks[x+1] > current_peak+(window_midpoint*2):
                single_sample_array.append([aligned_window, peaks[x]])
            else:
                single_sample_array.append([aligned_window, peaks[x]])
        else:
            single_sample_array.append([aligned_window, peaks[x]])

    return single_sample_array#, multiple_sample_array

def spike_location_accuracy(index_test, index_train, class_test):
    # difference = len(index_test) - len(index_train)
    all_indexes = []
    incorrect_indexes = []

    index_train_compare=index_train[:]
    i = 0
    for index in index_test:
        correct_flag = False
        for x in range(len(index_train_compare)):
            if abs(index-index_train_compare[x]) <= 50:
                correct_flag = True
                break
        if not correct_flag:
            incorrect_indexes.append([index, class_test[i]])
        else:
            index_train_compare.pop(0)
        
        all_indexes.append([index, class_test[i],correct_flag])
        i += 1

    success_rate = (len(index_test) - len(incorrect_indexes))/len(index_test)

    return incorrect_indexes, all_indexes, success_rate

mat = spio.loadmat('../neural-spike-sorting/datasets/training.mat', squeeze_me=True)

d = mat['d']

index_test, class_test = (list(t) for t in zip(*sorted(zip(mat['Index'], mat['Class']))))

samp_freq = 25000

filtered_d  = bandpass_filter(d, samp_freq)

smoothed_d = smoothing_filter(filtered_d)

time = list(np.linspace(0, len(d)*1/samp_freq, len(d)))

# individual_sample_time = 2.5e-3

peaks, energy_x, edo_threshold, filtered_threshold = spike_detector(smoothed_d)
index_train = list(peaks[0])

time_test = []
for index in index_test:
    time_test.append(time[int(index)])

incorrect_peaks, all_peaks, location_accuracy = spike_location_accuracy(index_test, index_train, class_test)

labels = [x[1] for x in all_peaks if x[2]]

print(location_accuracy)

peak_times = []
peak_d = []
for peak in index_train:
    peak_times.append(time[int(peak)])
    peak_d.append(smoothed_d[int(peak)])

good_spike_array = spike_extractor(smoothed_d, index_train) #, bad_spike_array
all_spike_array = good_spike_array #+ bad_spike_array

good_spike_samples = [x[0] for x in good_spike_array]
#bad_spike_samples = [x[0] for x in bad_spike_array]
all_samples = good_spike_samples #+ bad_spike_samples

# Apply min-max scaling
scaler = sk.preprocessing.MinMaxScaler()
scaled_spike_samples = scaler.fit_transform(all_samples)

# Do PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_spike_samples)

# Split training and test
# train_portion = 0.8
# train_length = math.ceil(train_portion * len(all_spike_array))
# train_data = pca_result[:train_length]
# test_data = pca_result[train_length:]

# train_label = labels[:train_length]
# test_label = labels[train_length:]
train_data, test_data, train_label, test_label = train_test_split(pca_result[:,0:2], labels, test_size=0.20)

## K Nearest Neighbours
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(train_data, train_label)

predictions = KNN.predict(test_data)

correct = sum([1 for compare in zip(test_label,predictions) if compare[0] == compare[1]])

score = correct/len(predictions)

print(correct)
print(len(predictions))
print(score)

# fig, ax = plt.subplots(4, 1)

# x_start = 0
# x_end = 0.5

# color = 'tab:red'
# ax[0].set_xlabel("Seconds")
# ax[0].set_ylabel("Amplitude (mV)", color=color)
# ax[0].plot(time, d, color)
# ax[0].scatter(time_test, d[index_test], color='black', marker='x', linewidths=1)
# ax[0].tick_params(axis='y', labelcolor=color)
# ax[0].set_xlim([x_start,x_end])

# color = 'tab:blue'
# ax[1].set_xlabel("Seconds")
# ax[1].set_ylabel("Amplitude (mV)", color=color)
# ax[1].plot(time, filtered_d, color)
# ax[1].tick_params(axis='y', labelcolor=color)
# ax[1].set_xlim([x_start,x_end])

# color = 'tab:orange'
# ax[2].set_xlabel("Seconds")
# ax[2].set_ylabel("Amplitude (mV)", color=color)
# ax[2].tick_params(axis='y', labelcolor=color)
# ax[2].plot(time, smoothed_d, color=color)
# ax[2].scatter(peak_times, peak_d, color='black', marker='x', linewidths=1)
# ax[2].plot([0,58], [filtered_threshold, filtered_threshold], color='purple')
# ax[2].set_xlim([x_start,x_end])

# color = 'tab:green'
# ax[3].set_xlabel("Seconds")
# ax[3].set_ylabel("Amplitude (mV)", color=color)
# ax[3].tick_params(axis='y', labelcolor=color)
# ax[3].plot(time, energy_x, color=color)
# ax[3].plot([0,58], [edo_threshold, edo_threshold], color='yellow')
# ax[3].set_xlim([x_start,x_end])


# # # Show the figure
# fig.tight_layout()
# # # plt.draw()

# fig2, ax2 = plt.subplots(1, 2)
# i = 0
# for wave in good_spike_samples:
#     # if i%100 == 0:
#     ax2[0].plot(wave)
#     i += 1
# i = 0
# for wave in bad_spike_samples:
#     # if i%100 == 0:
#     ax2[1].plot(wave)
#     i += 1

# Plot the 1st principal component aginst the 2nd and use the 3rd for color
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.scatter(pca_result[:, 0], pca_result[:, 1], c=pca_result[:, 2])
ax3.set_xlabel('1st principal component')
ax3.set_ylabel('2nd principal component')
ax3.set_title('first 3 principal components')
fig3.subplots_adjust(wspace=0.1, hspace=0.1)


# Plot the 1st principal component aginst the 2nd and use the 3rd for color
fig4, ax4 = plt.subplots(figsize=(8, 8))
ax4.scatter(test_data[:, 0], test_data[:, 1], c=predictions)
ax4.set_xlabel('1st principal component')
ax4.set_ylabel('2nd principal component')
ax4.set_title('Spike')
fig4.subplots_adjust(wspace=0.1, hspace=0.1)

plt.show()

print('fin')
import scipy.io as spio
# import scipy.stats as stat
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt 
import numpy as np

def filter_data(data, samp_freq, low=10, high=2000, order=2):
    nyq_freq = samp_freq/2

    low_band = low/nyq_freq
    high_band = high/nyq_freq

    b, a  = butter(order, [low_band, high_band], btype='band')

    filtered_data = lfilter(b, a, data)

    return filtered_data

def threshold_finder(filtered_data, thresh_factor):
    scaled_abs_signal = [abs(x)/0.6745 for x in filtered_data]
    sigma_n = np.median(scaled_abs_signal)
    thr = thresh_factor * sigma_n
    return thr

def env_diff_operator(data):
    N_start = len(data)
    

def spike_extractor(filtered_data, threshold, window_time=2.5e-3):

    # sample_datapoints = window_time * samp_freq
    window_datapoints = 60
    # threshold_indexes = np.where(filtered_data>threshold)

    # for thr in threshold_indexes:
    #     print(thr)
    peak_indices = find_peaks(filtered_data, threshold)
    return peak_indices

mat = spio.loadmat('../neural-spike-sorting/datasets/training.mat', squeeze_me=True)

d = mat['d']

samp_freq = 25000
filtered_d  = filter_data(d, samp_freq)

time = np.linspace(0, len(d)*1/samp_freq, len(d))
time = list(time)

thresh_factor = 5
individual_sample_time = 2.5e-3

threshold = threshold_finder(filtered_d, thresh_factor)

peaks = spike_extractor(filtered_d, threshold)

peak_times = []
peak_d = []
for peak in peaks[0]:
    peak_times.append(time[int(peak)])
    peak_d.append(filtered_d[int(peak)])


# Index = mat['Index']
# Class = mat['Class']

fig, ax = plt.subplots(2, 1)

plt.xlim(1,5)

color = 'tab:red'
ax[0].set_xlabel("Seconds")
ax[0].set_ylabel("Amplitude (mV)", color=color)
ax[0].plot(time, d, color)
ax[0].tick_params(axis='y', labelcolor=color)

# color = 'tab:black'
color = 'tab:blue'
ax[1].set_xlabel("Seconds")
ax[1].set_ylabel("Amplitude (mV)", color=color)
ax[1].plot(time, filtered_d, color)
ax[1].tick_params(axis='y', labelcolor=color)

ax[1].scatter(peak_times, peak_d, color='black', marker='x', linewidths=1)
ax[1].plot([0,58], [threshold, threshold], color='red')


# # Show the figure
fig.tight_layout()
# plt.draw()

plt.show()
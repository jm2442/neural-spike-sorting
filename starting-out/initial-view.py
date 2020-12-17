import scipy.io as spio
# import scipy.stats as stat
from scipy.signal import butter, lfilter, find_peaks, savgol_filter
import matplotlib.pyplot as plt 
import numpy as np

def bandpass_filter(data, samp_freq, low=10, high=10000, order=2):
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
    
def spike_extractor(filtered_data, window_time=2.5e-3):

    # sample_datapoints = window_time * samp_freq
    window_datapoints = 60
    # threshold_indexes = np.where(filtered_data>threshold)

    # for thr in threshold_indexes:
    #     print(thr)
    energy_x = envel_deriv_operator(filtered_data)

    thresh_factor = 5
    threshold = threshold_finder(energy_x, thresh_factor)
    threshold2 = threshold_finder(filtered_data,thresh_factor)

    print(threshold)
    peak_indices = find_peaks(energy_x, threshold)
    return peak_indices, energy_x, threshold, threshold2

def smoothing_filter(filtered_data):

    smoothed_data = savgol_filter(filtered_data, 27, 3)

    diff = filtered_data - smoothed_data
    return smoothed_data

mat = spio.loadmat('../neural-spike-sorting/datasets/training.mat', squeeze_me=True)

d = mat['d']

samp_freq = 25000

filtered_d  = bandpass_filter(d, samp_freq)

smoothed_d = smoothing_filter(filtered_d)

time = np.linspace(0, len(d)*1/samp_freq, len(d))
time = list(time)

individual_sample_time = 2.5e-3

peaks, energy_x, edo_threshold, filtered_threshold = spike_extractor(smoothed_d)

peak_times = []
peak_d = []
for peak in peaks[0]:
    peak_times.append(time[int(peak)])
    peak_d.append(smoothed_d[int(peak)])


# Index = mat['Index']
# Class = mat['Class']

fig, ax = plt.subplots(4, 1)

x_start = 0.3
x_end = 0.4

color = 'tab:red'
ax[0].set_xlabel("Seconds")
ax[0].set_ylabel("Amplitude (mV)", color=color)
ax[0].plot(time, d, color)
ax[0].tick_params(axis='y', labelcolor=color)
ax[0].set_xlim([x_start,x_end])

color = 'tab:blue'
ax[1].set_xlabel("Seconds")
ax[1].set_ylabel("Amplitude (mV)", color=color)
ax[1].plot(time, filtered_d, color)
ax[1].tick_params(axis='y', labelcolor=color)
ax[1].set_xlim([x_start,x_end])

color = 'tab:orange'
ax[2].set_xlabel("Seconds")
ax[2].set_ylabel("Amplitude (mV)", color=color)
ax[2].tick_params(axis='y', labelcolor=color)
ax[2].plot(time, smoothed_d, color=color)
# ax[2].plot([0,58], [threshold, threshold], color='yellow')
ax[2].scatter(peak_times, peak_d, color='black', marker='x', linewidths=1)
ax[2].plot([0,58], [filtered_threshold, filtered_threshold], color='purple')
ax[2].set_xlim([x_start,x_end])

color = 'tab:green'
ax[3].set_xlabel("Seconds")
ax[3].set_ylabel("Amplitude (mV)", color=color)
ax[3].tick_params(axis='y', labelcolor=color)
ax[3].plot(time, energy_x, color=color)
ax[3].plot([0,58], [edo_threshold, edo_threshold], color='yellow')
ax[3].set_xlim([x_start,x_end])


# # Show the figure
fig.tight_layout()
# plt.draw()

plt.show()
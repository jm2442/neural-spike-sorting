from scipy.signal import butter, lfilter, savgol_filter
from decimal import Decimal, ROUND_UP
import numpy as np

def bandpass(data, samp_freq, low=1, high=2000, order=2):
    nyq_freq = samp_freq/2
    low_band = low/nyq_freq
    high_band = high/nyq_freq
    b, a  = butter(order, [low_band, high_band], btype='band')
    filtered_data = lfilter(b, a, data)

    return filtered_data

def smoothing(filtered_data, window_size=17):

    new_window_size = int(Decimal(str(window_size)).quantize(Decimal('1.'), rounding=ROUND_UP))
    if new_window_size % 2 == 0:
        if new_window_size - window_size + 0.5 < 0:
            new_window_size += 1
        else:
            new_window_size -= 1

    smoothed_data = savgol_filter(filtered_data, int(round(new_window_size,0)), 3)

    return smoothed_data

def signal_processing(d, low_cutoff, high_cutoff, smooth_size):
    # Set the sampling frequency that is given to us, convert frequency sample into cumulative time
    samp_freq = 25000
    time = list(np.linspace(0, len(d)*1/samp_freq, len(d)))

    # Apply a bandpass and a Savitzky-golay filter to remove worst of the noise
    filt_d  = bandpass(d, samp_freq, low_cutoff, high_cutoff)
    smth_d = smoothing(filt_d, smooth_size)

    return time, filt_d, smth_d
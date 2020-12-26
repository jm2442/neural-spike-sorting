# Import libraries required
from scipy.signal import butter, lfilter, savgol_filter
from decimal import Decimal, ROUND_UP
import numpy as np

def bandpass(data, samp_freq, low=10, high=10000, order=2):
    # Returns a 1D signal that has been filtered with a bandpass filter

    # Calculate the nyquist frequency
    nyq_freq = samp_freq/2

    # Determine the numerator and the denominators of the IIR for the bandpass filter
    low_band = low/nyq_freq
    high_band = high/nyq_freq
    b, a  = butter(order, [low_band, high_band], btype='band')

    # Filter the data through the created bandpass filter
    filtered_data = lfilter(b, a, data)

    return filtered_data

def smoothing(filtered_data, window_size=17):
    # Returns a 1D signal that has been smoothed using a Savitzky-Golay filter

    # For the purpose of the optimisation, the window size is modifiable but it must be a odd integer and so the Decimal library has been used to ensure the float is rounder to a perfect int e.g. rather than stored as 16.9999999120
    new_window_size = int(Decimal(str(window_size)).quantize(Decimal('1.'), rounding=ROUND_UP))

    # When cast to an integer if the number is even decrement to an odd number
    if new_window_size % 2 == 0:
        new_window_size -= 1

    # Filter the data through the savitzky golay filter
    smoothed_data = savgol_filter(filtered_data, int(round(new_window_size,0)), 3)

    return smoothed_data

def signal_processing(d, low_cutoff, high_cutoff, smooth_size):
    # Returns the output from the signal processing operations that have been preformed

    # Set the sampling frequency that is given to us, convert frequency sample into cumulative time
    samp_freq = 25000
    time = list(np.linspace(0, len(d)*1/samp_freq, len(d)))

    # Apply a bandpass and a Savitzky-golay filter to remove worst of the noise
    filt_d  = bandpass(d, samp_freq, low_cutoff, high_cutoff)
    smth_d = smoothing(filt_d, smooth_size)

    return time, filt_d, smth_d
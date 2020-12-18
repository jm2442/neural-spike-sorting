from scipy.signal import butter, lfilter, savgol_filter

def bandpass(data, samp_freq, low=1, high=2000, order=2):
    nyq_freq = samp_freq/2
    low_band = low/nyq_freq
    high_band = high/nyq_freq
    b, a  = butter(order, [low_band, high_band], btype='band')
    filtered_data = lfilter(b, a, data)

    return filtered_data

def smoothing(filtered_data, window=17):
    smoothed_data = savgol_filter(filtered_data, window, 3)

    return smoothed_data
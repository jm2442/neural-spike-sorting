import numpy as np

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



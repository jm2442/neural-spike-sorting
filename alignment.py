import numpy as np

def spike_extractor(filtered_data, peaks, window_size=64):

    # Do initial extraction of spike according to peak from energy operator
    single_sample_array = []
    # multiple_sample_array = []
    window_midpoint = int(window_size)//2
    window_thirdpoint = int(window_size)//3

    for x in range(len(peaks)):

        current_peak = peaks[x]
        if current_peak - window_thirdpoint < 0: 
            left = 0
            left_zero_add = window_thirdpoint - current_peak
        else:
            left  = current_peak-5
            left_zero_add = 0

        if current_peak + (2*window_thirdpoint) > len(filtered_data): 
            right = len(filtered_data)-1
            right_zero_add = len(filtered_data) - current_peak
        else:
            right = current_peak+5
            right_zero_add = 0

        window = filtered_data[left:right]
        # window = left_zero_add*[0]
        # right_zero =  right_zero_add*[0]

        # window.extend(old_window)
        # window.extend(right_zero)


        max_point  = np.where(filtered_data==max(window))
        aligned_max = max_point[0][0]


        if aligned_max - window_thirdpoint < 0: 
            aligned_left = 0
        else:
            aligned_left  = aligned_max-window_thirdpoint

        if aligned_max + window_thirdpoint > len(filtered_data): 
            aligned_right = len(filtered_data)-1
        else:
            aligned_right = aligned_max+(2*window_thirdpoint)

        aligned_window = filtered_data[aligned_left:aligned_right]
        # aligned_window = left_zero_add*[0]
        # right_zero =  right_zero_add*[0]
        # aligned_window.extend(ali_window)
        # aligned_window.extend(right_zero)
        if len(aligned_window) < window_size:
            if left_zero_add > 1:
                left_zero = (window_size - len(aligned_window))*[0]
                left_zero.extend(aligned_window)
                aligned_window = np.array(left_zero[:])
            elif right_zero_add > 1:
                right_zero = (window_size - len(aligned_window))*[0]
                aligned_window = np.hstack([aligned_window, right_zero])


        if x+1 < len(peaks):
            if peaks[x-1] < current_peak-(window_midpoint*2) and peaks[x+1] > current_peak+(window_midpoint*2):
                single_sample_array.append([aligned_window, peaks[x]])
            else:
                single_sample_array.append([aligned_window, peaks[x]])
        else:
            single_sample_array.append([aligned_window, peaks[x]])

    return single_sample_array#, multiple_sample_array



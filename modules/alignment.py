# Import libraries required
import numpy as np

def spike_extractor(filtered_data, peaks, window_size):
    # Extracts the spikes according to a window around the identified peaks

    # List of windows contain spikes to be returned
    single_sample_array = []

    # Find the distance a third of the way from the window's edge
    window_thirdpoint = int(window_size)//3

    # Loop through the entire list of peaks found
    for x in range(len(peaks)):

        current_peak = peaks[x]

        # If the peak is early enough in the sample that the smaller window is too large, 
        if current_peak - window_thirdpoint < 0: 
            # Determine the left point of the smaller window and how many zeros will have to be added to the left
            left = 0
            left_zero_add = window_thirdpoint - current_peak
        else:
            # Otherwise, make the smaller window start 5 to the left of the peak
            left  = current_peak-5
            left_zero_add = 0

        # If the peak is late enough in the sample that the smaller window is too large, 
        if current_peak + (2*window_thirdpoint) > len(filtered_data): 
            # Determine the right point of the smaller window and how many zeros will have to be added on to the right
            right = len(filtered_data)-1
            right_zero_add = len(filtered_data) - current_peak
        else:
            # Otherwise, make the smaller window start 5 to the right of the peak
            right = current_peak+5
            right_zero_add = 0

        # Create the small window to act as a search area for the actual max point of the peak
        window = filtered_data[left:right]
        max_point  = np.where(filtered_data==max(window))
        aligned_max = max_point[0][0]

        # Determine the left point of the aligned window based off the new max
        if aligned_max - window_thirdpoint < 0: 
            aligned_left = 0
        else:
            aligned_left  = aligned_max-window_thirdpoint

        # Determine the right point of the aligned window based off the new max      
        if aligned_max + window_thirdpoint > len(filtered_data): 
            aligned_right = len(filtered_data)-1
        else:
            aligned_right = aligned_max+(2*window_thirdpoint)

        # Create the aligned window that will be added to the list 
        aligned_window = filtered_data[aligned_left:aligned_right]

        # If the window is smaller than the size it should be because of the early or late peaks        
        if len(aligned_window) < window_size:
            # Check which side zeros must be added to to correct to the right size and alignment
            if left_zero_add > 1:
                left_zero = (window_size - len(aligned_window))*[0]
                left_zero.extend(aligned_window)
                aligned_window = np.array(left_zero[:])
            elif right_zero_add > 1:
                right_zero = (window_size - len(aligned_window))*[0]
                aligned_window = np.hstack([aligned_window, right_zero])
            else:
                zero_add = window_size - len(aligned_window)
                while zero_add > 0:
                    if zero_add % 2 == 0:
                        aligned_window = np.hstack([[0], aligned_window])
                    else:
                        aligned_window = np.hstack([aligned_window, [0]])
                    zero_add -= 1

        # Add the window the list of spike samples as well as the spikes location
        single_sample_array.append([aligned_window, peaks[x]])

    return single_sample_array

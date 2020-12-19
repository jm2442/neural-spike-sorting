from scipy.signal import find_peaks
import numpy as np

import performance_metrics as metrics


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


def peak_detector(filtered_data, energy_threshold=25):
    energy_x = envel_deriv_operator(filtered_data)
    threshold = threshold_finder(energy_x, energy_threshold)
    thresh_factor = 5
    threshold2 = threshold_finder(filtered_data,thresh_factor)
    peak_indices = find_peaks(energy_x, threshold, prominence=1)

    return peak_indices, energy_x, threshold, threshold2

def peak_location_accuracy(index_test, index_train, class_test, print_on=True):
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

    loc_succ_rate = (len(index_test) - len(incorrect_indexes))/len(index_test)

    metrics.peak_location(incorrect_indexes, loc_succ_rate, print_on)

    return all_indexes, loc_succ_rate
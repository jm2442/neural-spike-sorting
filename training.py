# Import libraries required
import scipy.optimize as opt 
from modules import spike_sorter as spsrt
import math

def objective(x, clf_type, print_on, plot_on, evaluate):
    # A function that simply returns value from spike sorter as negative number in order to minimise the output
    return -spsrt.spike_sorter(x, clf_type, print_on, plot_on, evaluate=True)

def parameters(clf_type):
    # Returns the optimal parameters for each classifier

    # Signal processing parameters common to both classifier types
    params = {
            "low_cutoff": 10,#3.34,10
            "high_cutoff": 8000,#9158.98,8000
            "smooth_size": 21,#
            "edo_thresh_factor": 10,#20training/10testing
            "window_size": 90#64 #100
    }

    if clf_type == 2:
        # Optimal Params found by optimiser clf_type 2
        # "low_cutoff": 
        # "high_cutoff": 
        # "smooth_size": 
        # "edo_thresh_factor":
        # "window_size": 
        # "num_layers": 1.0
        # "num_neurons": 100.0
        # "act_function": 3.0
        # "alpha": 0.0001
        # "learn_rate_type": 0.0
        # learning rate = 0.01
        # hidden layer = 10
        # reLU
        # maxIter = 100

        params["num_layers"] = 1.0
        params["num_neurons"] = 100.0
        params["act_function"] = 3.0
        params["alpha"] = 0.0001
        params["learn_rate_type"] = 0.0

    elif clf_type == 3:
        # Optimal Params found by optimiser clf_type 3 - 95.1/95.86 95.11
        # "low_cutoff": 3.34/1.76   12.33
        # "high_cutoff": 9158.98/2615.15    1493.50
        # "smooth_size": 11 15
        # "edo_thresh_factor": 19.51    17.52
        # "window_size": 24 41
        # "num_neighbours": 5   #7
        # Euclidean
        # weight = distance

        ### PCA= 16

        params["num_neighbours"] = 7

    # Output parameters as list to pass to optimiser
    parameters = []
    for key, value in params.items(): 
        parameters.append(value)

    return parameters

def bounds(clf_type):
    # Returns the bounds for each parameter that can be tested within the optimiser

    # Signal processing bounds common to both classifier types
    b1 = (1, 50)
    b2 = (1000, 10000)
    b3 = (5, 39)
    b4 = (5, 100)
    b5 = (20, 100)
    bounds_pro = [b1, b2, b3, b4, b5]   

    if clf_type == 2:
        # Bounds for MLP-classifer specific parameters
        b6 = (1.0, 3.999)
        b7 = (22, 150)
        b8 = (0.0, 3.999)
        b9 = (0.0001, 0.1)
        b10 = (0.0, 2.999)
        bounds_class = [b6, b7, b8, b9, b10]

    elif clf_type == 3:
        # Bounds for KNN-classifer specific parameters
        b6 = (5, 50.99999)
        bounds_class = [b6]

    bounds = tuple(bounds_pro + bounds_class)

    return bounds

if __name__ == "__main__":
    
    # Set the fixed arguments which can be passed to the function
    clf_type = 3
    print_on = False
    plot_on = False
    evaluate = True
    args = (clf_type, print_on, plot_on, evaluate)

    # Toggle between running the optimiser or evaluating the training only once
    optimiser = False


    if optimiser:
        # Run the optimiser for a maximum of 25 iterations for time considerations
        result = opt.dual_annealing(objective, bounds=bounds(clf_type), maxiter=25, args=args) 
        print(result)
    else:
        # Evaluate the training of the current set parameters
        total_success = spsrt.spike_sorter(parameters(clf_type), clf_type, print_on=True, plot_on=True, evaluate=True)
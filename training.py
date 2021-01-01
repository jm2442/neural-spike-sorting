# Import libraries required
import scipy.optimize as opt 
from modules import spike_sorter as spsrt
import math

def objective(x, fixed_args, clf_type, print_on, plot_on, evaluate):
    # A function that simply returns value from spike sorter as negative number in order to minimise the output
    return -spsrt.spike_sorter(x, fixed_args, clf_type, print_on, plot_on, evaluate=True)

def parameters(clf_type):
    # Returns the optimal parameters for each classifier
    
    # Signal processing parameters common to both classifier types
    params = {
            "samp_freq": 25000,
            "low_cutoff": 10,
            "high_cutoff": 8000,
            "smooth_size": 21,
            "edo_thresh_factor": 20#20training/10testing
    }
    
    if clf_type == 2:
        # Optimal Params found by optimiser clf_type 2
        # "num_layers": 1.0
        # "num_neurons": 100.0
        params["num_layers"] = 1.0
        params["num_neurons"] = 100.0

    elif clf_type == 3:
        # Optimal Params found by optimiser clf_type 3 
        # "num_neighbours": 8
        params["num_neighbours"] = 8

    # Output parameters as list to pass to optimiser
    parameters = []
    for key, value in params.items(): 
        parameters.append(value)

    return parameters

def bounds(clf_type):
    # Returns the bounds for each parameter that can be tested within the optimiser

    if clf_type == 2:
        # Bounds for MLP-classifer specific parameters
        b6 = (1.0, 3.999)
        b7 = (22, 150)
        bounds_class = [b6, b7]

    elif clf_type == 3:
        # Bounds for KNN-classifer specific parameters

        b6 = (5, 50.99999)
        bounds_class = [b6]

    bounds = tuple(bounds_class)

    return bounds

def fixed_arguments(clf_type):
    # Returns the fixed arguments for each classifier
    args = {"window_size": 90}
    
    if clf_type == 2:
        #  fixed arguments for clf_type 2
        args["act_function"] = 'relu'
        args["alpha"] = 0.0001
        args["learn_rate_type"] = 'constant'
        args["learn_rate_init"] = 0.001
        args["max_iter"] = 100

    elif clf_type == 3:
        # fixed arguments for clf_type 3 
        args["pca_dim"] = 3
        args["weights"] = 'distance'


    # Output parameters as list to pass to optimiser
    arguments = []
    for key, value in args.items(): 
        arguments.append(value)

    return arguments

if __name__ == "__main__":
    
    # Set the fixed arguments which can be passed to the function
    clf_type = 2

    # Toggle between running the optimiser or evaluating the training only once
    optimiser = True

    if optimiser:
        print_on = False
        plot_on = False
        evaluate = True
        args = (fixed_arguments(clf_type), clf_type, print_on, plot_on, evaluate)
        # Run the optimiser for a maximum of 25 iterations for time considerations
        result = opt.dual_annealing(objective, bounds=bounds(clf_type), maxiter=25, args=args) 
        print(result)
    else:
        # Evaluate the training of the current set parameters
        total_success = spsrt.spike_sorter(parameters(clf_type), fixed_arguments(clf_type), clf_type, print_on=True, plot_on=True, evaluate=True)
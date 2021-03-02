# Import libraries required
import scipy.optimize as opt 
from code import spike_sorter as spsrt
import math

def objective(x, fixed_args, clf_type, print_on, plot_on, evaluate, x_start, x_end):
    # A function that simply returns value from spike sorter as negative number in order to minimise the output
    return -spsrt.spike_sorter(x, fixed_args, clf_type, print_on, plot_on, evaluate, x_start, x_end)

def parameters(clf_type):
    # Returns the optimal parameters for each classifier
    
    # Signal processing parameters common to both classifier types
    params = {                      
            "low_cutoff": 10,       
            "high_cutoff": 8000,   
            "smooth_size": 21,      
            "edo_thresh_factor": 10
    #Initial    #Opt 1      #Opt 2      #Adjusted
    #92.99      #93.97      #95.58      #93.46
    #10         #4.68       #1.77       #10
    #10000      #3379.49    #1623.61    #8000
    #11         #19         #21         #21    
    #10         #20         #13         #10
    #5          #10         #5          #7  
    }
    
    if clf_type == 'MLP':
        # Optimal Params found by optimiser MLP
        params["num_layers"] = 1.0
        params["num_neurons"] = 100.0

    elif clf_type == 'KNN':
        # Optimal Params found by optimiser KNN 
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
    bounds_pro = [b1, b2, b3, b4]  

    if clf_type == 'MLP':
        # Bounds for MLP-classifer specific parameters
        b6 = (1.0, 3.999)
        b7 = (22, 150)
        bounds_class = [b6, b7]

    elif clf_type == 'KNN':
        # Bounds for KNN-classifer specific parameters

        b6 = (5, 50.99999)
        bounds_class = [b6]

    bounds = tuple(bounds_pro + bounds_class)

    return bounds

def fixed_arguments(clf_type):
    # Returns the fixed arguments for each classifier
    args = {
        "samp_freq": 25000,
        "window_size": 90
        }

    if clf_type == 'MLP':
        #  fixed arguments for clf_type 2
        args["act_function"] = 'relu'
        args["alpha"] = 0.0001
        args["learn_rate_type"] = 'constant'
        args["learn_rate_init"] = 0.001
        args["max_iter"] = 100

    elif clf_type == 'KNN':
        # fixed arguments for clf_type 3 
        args["pca_dim"] = 3
        args["weights"] = 'distance'


    # Output parameters as list to pass to optimiser
    arguments = []
    for key, value in args.items(): 
        arguments.append(value)

    return arguments

if __name__ == "__main__":

    ########## INPUTS ##########
    # Set the Classifier of choice. (MLP) or (KNN)
    clf_type = 'KNN' 

    # Toggle between running the optimiser or evaluating the training only once
    optimiser = False

    # Set the boundaries for timeseries plots
    x_start = 0
    x_end = 0.08
    ############################

    if optimiser:
        print_on = False
        plot_on = False
        evaluate = True
        args = (fixed_arguments(clf_type), clf_type, print_on, plot_on, evaluate, x_start, x_end)
        # Run the optimiser for a maximum of 25 iterations for time considerations
        result = opt.dual_annealing(objective, bounds=bounds(clf_type), maxiter=25, args=args) 
        print(result)
    else:
        print_on = True
        plot_on = True
        evaluate = True
        # Evaluate the training of the current set parameters
        total_success = spsrt.spike_sorter(parameters(clf_type), fixed_arguments(clf_type), clf_type, print_on, plot_on, evaluate, x_start, x_end)
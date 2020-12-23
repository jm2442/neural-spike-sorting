import scipy.optimize as opt 
import spike_sorting as spsrt
import math

def objective(x):#, args):
    return -spsrt.spike_sorter(x)#, args)

part = 3
# params = {}
params = {#}
        "low_cutoff": 3.34,
        "high_cutoff": 9158.98,
        "smooth_size": 11,
        "edo_thresh_factor": 19.51,
        "window_size": 30#24
}

b1 = (1, 50)
b2 = (1000, 10000)
b3 = (5, 39)
b4 = (5, 100)
b5 = (20, 100)

bounds_pro = [b1, b2, b3, b4, b5]

if part == 2:
    # Optimal Params found by optimiser PART 2
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

    params["num_layers"] = 1.0
    params["num_neurons"] = 100.0
    params["act_function"] = 3.0
    params["alpha"] = 0.0001
    params["learn_rate_type"] = 0.0

    b6 = (1.0, 3.999)
    b7 = (22, 150)
    b8 = (0.0, 3.999)
    b9 = (0.0001, 0.1)
    b10 = (0.0, 2.999)

    bounds_class = [b6, b7, b8, b9, b10]

elif part == 3:
    # Optimal Params found by optimiser PART 3 - 95.12
    # "low_cutoff": 3.34
    # "high_cutoff": 9158.98
    # "smooth_size": 11
    # "edo_thresh_factor": 19.51
    # "window_size": 24
    # "num_neighbours": 5

    params["num_neighbours"] = 5

    b6 = (5, 50.99999)

    bounds_class = [b6]

bounds = tuple(bounds_pro + bounds_class)


x0 = []
for key, value in params.items(): 
    x0.append(value)

# args = (part,)

total_success = spsrt.spike_sorter(x0)#, args)

# result = opt.dual_annealing(objective, bounds=bounds, maxiter=100) #, args=args # , x0=x0
# print(result)
# print("finished")
# for res in result.x:
#     print(res)

# result = opt.minimize(objective, x0, args=args ,method='BFGS', constraints=constraints, bounds=bounds, options={'disp':True})
# result = opt.differential_evolution(objective, args=args, bounds=bounds, popsize=50)
# result = opt.basinhopping(objective, args=args, x0=x0)#, bounds=bounds)




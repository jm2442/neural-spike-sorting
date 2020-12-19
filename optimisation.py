import scipy.optimize as opt 
import spike_sorting as spsrt
import math

def objective(x):
    return -spsrt.spike_sorter(x)

def constraint1(x):
    if int(round(x[2],0)) % 2 == 0.0:
        ans = -1
    else:
        ans = 1
    return ans

params = {
    "low_cutoff": 10,
    "high_cutoff": 1000,
    "smooth_size": 39,
    "edo_thresh_factor": 6,
    "window_size": 21,
    "num_neighbours": 5,
}

b1 = (1, 50)
b2 = (1000, 10000)
b3 = (5, 39)
b4 = (5, 100)
b5 = (20, 100)
b6 = (5, 50)

bounds = (b1, b2, b3, b4, b5, b6)
con1 = {"type": "ineq", "fun": constraint1}
constraints = [con1]

x0 = []
for key, value in params.items(): 
    x0.append(value)

# total_success = spsrt.spike_sorter(x0)

print(params)
# result = opt.minimize(objective, x0, method='BFGS', constraints=constraints, bounds=bounds, options={'disp':True})
# result = opt.dual_annealing(objective, x0=x0, bounds=bounds)
result = opt.basinhopping(objective, x0=x0)#, bounds=bounds)
print(result)
import scipy.io as spio

mat = spio.loadmat('neural-spike-sorting/datasets/training.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
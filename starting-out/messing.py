import scipy.io as spio
import matplotlib.pyplot as plt 

mat = spio.loadmat('../neural-spike-sorting/datasets/training.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
print('HelloWorld')
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel("")
ax1.set_ylabel("Amplitude (mV)", color=color)
ax1.plot(d, color)
ax1.tick_params(axis='y', labelcolor=color)

# Show the figure
fig.tight_layout()
plt.show()
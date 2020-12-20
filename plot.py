import matplotlib.pyplot as plt
import numpy as np

def filter_and_detection(x_start, x_end, time, data, time_test, index_train, index_test, filtered_data, smoothed_data, smoothed_threshold, edo_data, edo_threshold):


    peak_times = [time[int(peak)] for peak in index_train]
    peak_data = [smoothed_data[int(peak)] for peak in index_train]

    fig, ax = plt.subplots(2, 1)

    # Plot Original Wave
    color = 'tab:red'
    ax[0].set_xlabel("Seconds")
    ax[0].set_ylabel("Amplitude (mV)", color=color)
    ax[0].plot(time, data, color)
    ax[0].scatter(time_test, data[index_test], color='black', marker='x', linewidths=1)
    ax[0].tick_params(axis='y', labelcolor=color)
    ax[0].set_xlim([x_start,x_end])

    # Plot Bandpass Output
    color = 'tab:blue'
    ax[0].plot(time, filtered_data, color)

    # Plot Savitzky-Golay Filter Output
    color = 'tab:orange'
    ax[1].set_xlabel("Seconds")
    ax[1].set_ylabel("Amplitude (mV)", color=color)
    ax[1].tick_params(axis='y', labelcolor=color)
    ax[1].plot(time, smoothed_data, color=color)
    ax[1].scatter(peak_times, peak_data, color='black', marker='x', linewidths=1)
    ax[1].plot([0,58], [smoothed_threshold, smoothed_threshold], color='purple')
    ax[1].set_xlim([x_start,x_end])

    # Plot EDO Output
    color = 'tab:green'
    ax[1].plot(time, edo_data, color=color)
    ax[1].plot([0,58], [edo_threshold, edo_threshold], color='yellow')

    fig.tight_layout()
    plt.draw()

def samples(data_samples, interval=1):

    fig, ax = plt.subplots()
    i = 0
    for wave in data_samples:
        if i%interval == 0:
            ax.plot(wave)
    i += 1
    fig.tight_layout()
    plt.draw()

def PCA(pca):

    # Plot the 1st principal component aginst the 2nd and use the 3rd for color
    fig, ax = plt.subplots()
    ax.scatter(pca[:, 0], pca[:, 1], c=pca[:, 2])

    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.set_title('first 3 principal components')
    
    fig.tight_layout()
    plt.draw()

def KNN(test_data, prediction_label, data_samples):

    # Sort each wave sample into its corresponding class type
    cluster_list=[1, 2, 3, 4]
    for cluster in cluster_list:
        cluster_list[cluster-1] = [data_samples[x] for x in range(len(test_data)) if prediction_label[x] == cluster]

    # Plot the 1st principal component aginst the 2nd and use the 3rd for color
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(test_data[:, 0], test_data[:, 1], c=prediction_label)
    ax[0].set_xlabel('1st principal component')
    ax[0].set_ylabel('2nd principal component')
    ax[0].set_title('Spike')

    time = range(int(np.size(cluster_list[0][0])))
    for i in range(4):
        clust_mean = np.array(cluster_list[i]).mean(axis=0)
        clust_std = np.array(cluster_list[i]).std(axis=0)

        ax[1].plot(time, clust_mean, label='Neuron {}'.format(i))
        ax[1].fill_between(time, clust_mean-clust_std, clust_mean+clust_std, alpha=0.15)

    ax[1].set_title('average waveforms')
    ax[1].legend()
    ax[1].set_xlabel('time [ms]')
    ax[1].set_ylabel('amplitude [uV]')

    fig.tight_layout()
    plt.draw()
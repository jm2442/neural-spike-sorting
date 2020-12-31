# Import libraries required
from sklearn import metrics as skmetrics
import matplotlib.pyplot as plt
import numpy as np

def filter_and_detection(x_start, x_end, time, data, time_test, index_train, index_test, filtered_data, smoothed_data, smoothed_threshold, edo_data, edo_threshold, training=True):
    # Plots the effect of the filters and peak detection of the input signal

    # Extract the signal values and the timing of peaks to plot the identified peaks
    peak_times = [time[int(peak)] for peak in index_train]
    peak_data = [smoothed_data[int(peak)] for peak in index_train]

    fig, ax = plt.subplots(4, 1)

    # Plot Original Wave
    color = 'tab:red'
    ax[0].set_title("Signal Processing & Peak Detection")
    # ax[0].set_xlabel("Seconds")
    # ax[0].set_ylabel("Amplitude (mV)")
    ax[0].plot(time, data, color, label='Original Signal')
    if training:
        ax[0].scatter(time_test, data[index_test], color='black', marker='x', linewidths=1)
    # ax[0].tick_params(axis='y', labelcolor=color)
    ax[0].set_xlim([x_start,x_end])
    # Plot Bandpass Output
    color = 'tab:blue'
    ax[0].plot(time, filtered_data, color, label='Bandpass Filtered Signal')
    ax[0].get_xaxis().set_visible(False)
    ax[0].legend()

    # Plot Bandpass Output
    color = 'tab:blue'
    ax[1].plot(time, filtered_data, color, label='Bandpass Filtered Signal')
    # Plot Savitzky-Golay Filter Output
    color = 'tab:orange'
    # ax[1].set_xlabel("Seconds")
    # ax[1].set_ylabel("Amplitude (mV)")
    # ax[1].tick_params(axis='y', labelcolor=color)
    ax[1].plot(time, smoothed_data, color=color, label='Sav-Gol Filtered Signal')
    ax[1].set_xlim([x_start,x_end])
    ax[1].get_xaxis().set_visible(False)
    ax[1].legend()

    # Plot Savitzky-Golay Filter Output
    color = 'tab:orange'
    # ax[2].set_xlabel("Seconds")
    # ax[2].set_ylabel("Amplitude (mV)")
    # ax[2].tick_params(axis='y', labelcolor=color)
    ax[2].plot(time, smoothed_data, color=color, label='Sav-Gol Filtered Signal')
    ax[2].plot([0,58], [smoothed_threshold, smoothed_threshold], color='purple', label='5*MAD of Sav-Gol')
    # Plot EDO Output
    color = 'tab:green'
    ax[2].plot(time, edo_data, color=color, label='EDO of Signal')
    ax[2].plot([0,58], [edo_threshold, edo_threshold], color='yellow', label='Threshold Const. * MAD of EDO')
    ax[2].set_xlim([x_start,x_end])
    ax[2].set_ylim([-5,50])
    ax[2].get_xaxis().set_visible(False)
    ax[2].legend()

    # Plot Savitzky-Golay Filter Output
    color = 'tab:orange'
    ax[3].set_xlabel("Seconds")
    # ax[3].set_ylabel("Amplitude (mV)")
    # ax[3].tick_params(axis='y', labelcolor=color)
    ax[3].plot(time, smoothed_data, color=color, label='Sav-Gol Filtered Signal')
    # Plot detected peaks
    ax[3].scatter(peak_times, peak_data, color='black', marker='x', linewidths=1, label='Detected Spikes')
    ax[3].set_xlim([x_start,x_end])
    ax[3].legend()

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.draw()

def samples(data_samples, interval=1):
    # Plots the extract spike samples at a set interval

    fig, ax = plt.subplots()
    i = 0
    for wave in data_samples:
        if i % interval == 0:
            ax.plot(wave[0], linewidth= 0.5, color='k', alpha =0.2)
    i += 1
    fig.tight_layout()
    plt.draw()

def PCA(pca):

    # Plots the 1st principal component aginst the 2nd and use the 3rd for color
    fig, ax = plt.subplots()
    ax.scatter(pca[:, 0], pca[:, 1], c=pca[:, 2])

    ax.set_xlabel('1st Principal Component')
    ax.set_ylabel('2nd Principal Component')
    ax.set_title('First 3 Principal Components')
    
    fig.tight_layout()
    plt.draw()

def KNN(test_data, prediction_label, data_samples, interval=1):

    # Sort each wave sample into its corresponding class type
    cluster_list=[0]*4
    for cluster in [1,2,3,4]:
        cluster_list[cluster-1] = [data_samples[x] for x in range(len(test_data)) if prediction_label[x] == cluster]

    # Create a window array based on the number of data points
    time = range(int(np.size(cluster_list[0][0])))

    # Plot the samples of each cluster at a set interval
    fig, ax = plt.subplots(1,4)
    j = 0
    k = 0
    for wave_cluster in cluster_list:
        for wave in wave_cluster:
            if k % interval == 0:
                ax[j].plot(wave, linewidth= 0.5, color='k', alpha =0.2)
                ax[j].set_xlabel('Spike '+ str(j+1))
            k += 1
        ax[j].set_ylim([-2,12])
        j += 1
    ax[0].set_ylabel('Amplitude (mV)')
    
    fig.tight_layout()
    plt.draw()

    #Set colours so that averaged wave matches cluster colour
    colourmap = np.array(['#1f77b4','#ff7f0e','#2ca02c','#d62728'])
    colour_pred= np.array([x-1 for x in prediction_label])

    # Plot the output of the KNN with clusters in corresponding colours
    pca = np.array(test_data)
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(pca[:, 0], pca[:, 1], c=colourmap[colour_pred])
    ax[0].set_xlabel('1st Principal Component')
    ax[0].set_ylabel('2nd Principal Component')
    ax[0].set_title('KNN Clustered Data')

    # Plot the mean of each cluster type and an outline of the standard deviation range
    for i in range(4):
        clust_mean = np.array(cluster_list[i]).mean(axis=0)
        clust_std = np.array(cluster_list[i]).std(axis=0)

        ax[1].plot(time, clust_mean, label='Spike {}'.format(i+1))
        ax[1].fill_between(time, clust_mean-clust_std, clust_mean+clust_std, alpha=0.15)

    ax[1].set_title('Average Waveforms')
    ax[1].legend()
    ax[1].set_xlabel('Sample Point')
    ax[1].set_ylabel('Amplitude (mV)')

    fig.tight_layout()
    plt.draw()

def MLP(test_data, prediction_label, data_samples, interval=1):

    # Sort each wave sample into its corresponding class type
    cluster_list=[0]*4
    for cluster in [1,2,3,4]:
        cluster_list[cluster-1] =  [test_data[x] for x in range(len(test_data)) if prediction_label[x] == cluster]

    # Create a window array based on the number of data points
    time = list(range(int(np.size(cluster_list[0][0]))))

    # Plot the samples of each cluster at a set interval
    fig, ax = plt.subplots(1,4)
    j = 0
    k = 0
    for wave_cluster in cluster_list:
        for wave in wave_cluster:
            if k % interval == 0:
                ax[j].plot(wave, linewidth= 0.5, color='k', alpha =0.2)
                ax[j].set_xlabel('Spike '+ str(j+1))
            k += 1
        ax[j].set_ylim([-2,12])
        j += 1
    ax[0].set_ylabel('Amplitude (mV)')

    fig.tight_layout()
    plt.draw()

    # Plot the mean of each cluster type and an outline of the standard deviation range
    fig, ax = plt.subplots(1,1)
    for i in range(4):
        clust_mean = np.array(cluster_list[i]).mean(axis=0)
        clust_std = np.array(cluster_list[i]).std(axis=0)

        ax.plot(time, clust_mean, label='Spike {}'.format(i))
        ax.fill_between(time, clust_mean-clust_std, clust_mean+clust_std, alpha=0.15)

    ax.set_title('average waveforms')
    ax.legend()
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('amplitude [uV]')

    fig.tight_layout()
    plt.draw()

def confusion_matrix(classifier, X_test, y_test):
    # Uses the sk learn library to plot a confusion matrix
    # fig, ax = plt.subplots(1, 1)
    disp = skmetrics.plot_confusion_matrix(classifier, X_test, y_test, cmap='cividis')
    # disp.ax_set_title('Confusion Matrix for Spike Classification')
    disp.ax_.set_title("Spike Classification Confusion Matrix")
    disp.figure_.tight_layout()
    plt.draw()

def spike_train(x_start, x_end, time, data, pred):
    # Takes the predicted output from a classifier and plots the location of the different outputs in the original signal

    # Loop the output and organise the location and values of the different neuron types
    neuron_types = [1,2,3,4]
    spike_trains = []
    for neuron in neuron_types:

        neuron_peaks = [x[1] for x in pred if x[0] == neuron]

        peak_times = [time[int(peak)] for peak in neuron_peaks]
        peak_data = [data[int(peak)] for peak in neuron_peaks]

        peak_lines = np.zeros(len(peak_times)) + (neuron*0.25)
        peak_place = peak_lines + max(peak_data)

        spike_trains.append([peak_times, peak_place])

    fig, ax = plt.subplots(1, 1)

    # Plot Original Wave
    color = 'tab:red'
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Amplitude (mV)")
    ax.plot(time, data, color)
    # ax.tick_params(axis='y', labelcolor=color)
    ax.set_xlim([x_start,x_end])

    # Calcute number of individual neurons
    spike_train_lens = [len(x[0]) for x in spike_trains]

    # Plot the classifier's predicted spikes
    points=['black','blue','green','purple']
    i = 0
    for spike in spike_trains:
        ax.scatter(spike[0], spike[1], color=points[i], marker='v', linewidths=1,label='Spike {}'.format(neuron_types[i]))
        i += 1

    ax.set_ylim([-2.5,16])
    plt.legend()

    fig.tight_layout()
    plt.draw()

    return spike_train_lens
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def filter_and_detection(x_start, x_end, time, data, time_test, index_train, index_test, filtered_data, smoothed_data, smoothed_threshold, edo_data, edo_threshold, training=True):


    peak_times = [time[int(peak)] for peak in index_train]
    peak_data = [smoothed_data[int(peak)] for peak in index_train]

    fig, ax = plt.subplots(2, 1)

    # Plot Original Wave
    color = 'tab:red'
    ax[0].set_xlabel("Seconds")
    ax[0].set_ylabel("Amplitude (mV)", color=color)
    ax[0].plot(time, data, color)
    if training:
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

    # Plot EDO Output
    color = 'tab:green'
    ax[1].plot(time, edo_data, color=color)
    ax[1].plot([0,58], [edo_threshold, edo_threshold], color='yellow')

    # Plot detected peaks
    ax[1].scatter(peak_times, peak_data, color='black', marker='x', linewidths=1)
    ax[1].plot([0,58], [smoothed_threshold, smoothed_threshold], color='purple')
    ax[1].set_xlim([x_start,x_end])


    fig.tight_layout()
    plt.draw()

def samples(data_samples, interval=1):

    fig, ax = plt.subplots()
    i = 0
    for wave in data_samples:
        if i % interval == 0:
            ax.plot(wave[0])
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
    cluster_list=[0]*4
    for cluster in [1,2,3,4]:
        temp = [data_samples[x] for x in range(len(test_data)) if prediction_label[x] == cluster]

        cluster_list[cluster-1] = temp
    gogo = np.array(test_data)
    # Plot the 1st principal component aginst the 2nd and use the 3rd for color

    time = range(int(np.size(cluster_list[0][0])))

    
    fig, ax = plt.subplots(1,4)
    j = 0
    k = 0
    for wave_cluster in cluster_list:
        for wave in wave_cluster:
            # if k % 20 == 0:
            ax[j].plot(time, wave)
            k += 1
        j += 1

    fig.tight_layout()
    plt.draw()


    fig, ax = plt.subplots(1,2)
    ax[0].scatter(gogo[:, 0], gogo[:, 1], c=prediction_label)
    ax[0].set_xlabel('1st principal component')
    ax[0].set_ylabel('2nd principal component')
    ax[0].set_title('Spike')

    for i in range(4):
        clust_mean = np.array(cluster_list[i]).mean(axis=0)
        clust_std = np.array(cluster_list[i]).std(axis=0)

        ax[1].plot(time, clust_mean, label='Neuron {}'.format(i+1))
        ax[1].fill_between(time, clust_mean-clust_std, clust_mean+clust_std, alpha=0.15)

    ax[1].set_title('average waveforms')
    ax[1].legend()
    ax[1].set_xlabel('time [ms]')
    ax[1].set_ylabel('amplitude [uV]')

    fig.tight_layout()
    plt.draw()

def MLP(test_data, prediction_label, data_samples):
    # Sort each wave sample into its corresponding class type
    cluster_list=[0]*4
    for cluster in [1,2,3,4]:
        temp = [test_data[x] for x in range(len(test_data)) if prediction_label[x] == cluster]

        cluster_list[cluster-1] = temp
        
    # Plot the 1st principal component aginst the 2nd and use the 3rd for color

    time = list(range(int(np.size(cluster_list[0][0]))))

    
    fig, ax = plt.subplots(1,4)
    j = 0
    k = 0
    for wave_cluster in cluster_list:
        for wave in wave_cluster:
            # if k % 20 == 0:
            ax[j].plot(time, wave)
            k += 1
        j += 1

    fig.tight_layout()
    plt.draw()


    fig, ax = plt.subplots(1,1)
    for i in range(4):
        clust_mean = np.array(cluster_list[i]).mean(axis=0)
        clust_std = np.array(cluster_list[i]).std(axis=0)

        ax.plot(time, clust_mean, label='Neuron {}'.format(i))
        ax.fill_between(time, clust_mean-clust_std, clust_mean+clust_std, alpha=0.15)

    ax.set_title('average waveforms')
    ax.legend()
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('amplitude [uV]')

    fig.tight_layout()
    plt.draw()

def confusion_matrix(classifier, X_test, y_test):
    # metrics.plot_confusion_matrix(classifier, X_test, y_test)
    # plt.draw()
    pass

def spike_train(x_start, x_end, time, data, pred):

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
    ax.set_ylabel("Amplitude (mV)", color=color)
    ax.plot(time, data, color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_xlim([x_start,x_end])

    # Plot detected peaks
    points=['black','blue','green','purple']
    i = 0
    for spike in spike_trains:
        ax.scatter(spike[0], spike[1], color=points[i], marker='v', linewidths=1,label='Neuron {}'.format(neuron_types[i]))
        i += 1
    # ax[0].set_xlim([x_start,x_end])
    plt.legend()

    fig.tight_layout()
    plt.draw()
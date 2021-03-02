# Neural Spike Sorting
A system to automatically analyse a set of recordings that have been made using a simple bipolar electrode inserted within the cortical region of the human brain.

This system has been developed using two different datasets of recordings. 

The first is a training dataset which has been generated using a detailed simulation model and contains a single time domain recording of the spikes from four types of neurons. Accompanying the recording are labels which identify the index and class of the spike within the dataset. The training dataset is used to build and evaluate the performance of the spike detection and classification using k-fold cross validation before its application to the submission dataset.

The second is a submission dataset which contains a real recording made from the cortical region of the brain. It contains the same spikes from the four neurons as in the training dataset. The submission dataset's noise is much worse than the training dataset as the subject was moving when the recordings were made. The output that are produced by the system are a .mat file that contains the respective indexes and classes of the spikes identified using the model on the submission dataset.

EE40098 – Coursework C,
Set by Dr B W Metcalfe,
Department of Electronic & Electrical Engineering,
University of Bath

# Instructions For Use
The code has been designed to have two top level scripts which call on functions from other files within the modules package.

1. **Training.py**\
The top level script for building, training and evaluating the spike sorting system based on the *training.mat* dataset provided.
The three main inputs for the script are:
      * Classifier type (clf_type)
          * 'MLP' - Multi-Layer Perceptron
          * 'KNN' - K Nearest Neighbours (with PCA) 
      * One off or Optimisation (optimiser)
          * True - System is to have its parameters tuned with an optimisation technique
          * False - System is to be evaluated with only the current parameters
      * X axis limits (i.e. the time interval) for time series plots (x_start and x_end) - Only used when plot_on = True

If more control over the specific inputs e.g. low cut-off frequency, number of PCA dimensions etc. for either the training or submission datasets is required, they can be accessed within the parameters() and fixed_arguments() functions.

2. **Submission.py**\
The top level script for building and training (based on the parameters set in *training.py*) the spike sorting system on the *training.mat* dataset and then evaluating the system performance on the *submission.mat* dataset provided by producing an output *submission.mat* file for examiner evaluation. Within *submission.py* The three main toggleable inputs for the script are:
      * Classifier type (clf_type)
          * 'MLP' - Multi-Layer Perceptron
          * 'KNN' - K Nearest Neighbours (with PCA) 
      * Plot output (plot_on)
      * X axis limits (i.e. the time interval) for time series plots (x_start and x_end) - Only used when plot_on = True

## Relation to Assignment tasks
1. Performance Metrics - modules/performance_metrics.py
2. MLP - modules/classification.py
3. KNN - modules/classification.py, PCA - modules/feature_extract_reduce.py
4. Dual Annealing Optimisation technique - training.py, Application to submission dataset - submission.py

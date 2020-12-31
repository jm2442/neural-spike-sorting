# Neural Spike Sorting
A system to automatically analyse a set of recordings that have been made from the human brain. 
EE40098 – Coursework C,
Set by Dr B W Metcalfe,
Department of Electronic & Electrical Engineering,
University of Bath

# Instructions For Use
The code has been designed to have two top level scripts which call on modules from other files within a package.

1. **Training.py**\
The top level script for building, training and evaluating the spike sorting system based on the *training.mat* dataset provided.
The two main toggleable inputs for the script are:
      * Classifier type (clf_type)
          * 2 - Multi-Layer Perceptron
          * 3 - K Nearest Neighbours (with PCA) 
      * One off or Optimisation (optimiser)
          * True - System is to have its parameters tuned with an optimisation technique
          * False - System is to be evaluated with current parameters

2. **Submission.py**\
The top level script for building and training (based on the parameters set in *training.py*) the spike sorting system on the *training.mat* dataset and then evaluating the system performance on the *submission.mat* dataset provided by producing an output *submission.mat* file for examiner evaluation. The two main toggleable inputs for the script are:
      * Classifier type (clf_type)
          * 2 - Multi-Layer Perceptron
          * 3 - K Nearest Neighbours (with PCA) 
      * Plot output (plot_on)

## Relation to Assignment tasks
1. Performance Metrics - modules/performance_metrics.py
2. MLP - modules/classification.py
3. KNN - modules/classification.py & PCA - modules/feature_extract_reduce.py
4. Dual Annealing Optimisation technique - training.py

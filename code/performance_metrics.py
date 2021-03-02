# Import libraries required
from sklearn import metrics

def peak_location(incorrect_indexes, success_rate, print_on=True):

    if print_on:
        # Print every 20th incorrect peak
        print("*"*20)
        print("Sample of Incorrect Indexes")
        i = 0
        for index in incorrect_indexes:
            if i % 20 == 0:
                print(f"{index[0]} - {index[0]/25000}")
            i += 1
        
        # Print the F1 score for determining the spike location
        print("*"*20)
        print(f"Location F1 Score (%) = {round(success_rate*100, 2)}")

def peak_location_accuracy(index_test, index_train, class_test, print_on=True):
    # Returns all the indexes including which of those have been correctly located, as well as the F1 score

    all_indexes = []
    incorrect_indexes = []

    index_test_compare = index_test[:]
    index_train_compare = index_train[:]

    # Keep a record between the differences between the identified peaks and the given index of spikes since these are located at the beginning of the spike rather than the maximum
    offsets = []
    for k in range(len(index_test_compare)):
        diff = abs(index_test_compare[k] - index_train_compare[k])
        if diff <= 50:
            offsets.append(diff)
        else:
            break

    # Loop through the given indexes for the peak locations
    i = 0
    TP = 0
    FN = 0
    for y in range(len(index_test_compare)):

        index = index_test_compare[y]
        correct_flag = False

        # Loop through the found indexes for the peak locations
        for x in range(len(index_train_compare)):

            ind_comp= index_train_compare[x]

            # If the difference between the peaks is less than the given threshold of 50, then a peak has been correctly identified (TP)
            diff = abs(index-ind_comp)
            if diff <= 50:

                # TRUE POSITIVE
                TP += 1

                # Keep track of avg difference
                offsets.append(diff)
                k += 1

                # Set correct flag true and break out of found peak loop
                correct_flag = True
                break

        if not correct_flag:
            # If no peak with a value within the threshold has been found, a peak has not been found (FN)

            # FALSE NEGATIVE
            FN += 1

            # Calculate running avg offset
            avg_offset = int(round(sum(offsets)/k, 0))

            # Record peak not found
            incorrect_indexes.append([index, class_test[i]])

            # Add the offset to give a better representation of where the peaks location would be
            all_indexes.append([index+avg_offset, class_test[i],correct_flag])
        else:
            # Record and remove found peak so that is cannot be found again
            all_indexes.append([ind_comp, class_test[i],correct_flag])
            index_train_compare.pop(0)
        
        i += 1

    # If anything is still left in the found peak lists, these are peaks that have been identified incorrectly (FP)

    # FALSE POSITIVE
    FP = len(index_train_compare)

    # TRUE NEGATIVE
    # All other mins total guesses, calculated for completeness sake
    # TN = num_occur - TP - FP - FN
    # if TN < 0 and len(index_train) < len(index_test): TN = 0

    # Manually calculate the precision, recall and F1 score
    precision = TP / float(TP + FP)
    recall = TP/ float(TP + FN)
    F1_score = 2*(recall * precision)/ (recall + precision)

    peak_location(incorrect_indexes, F1_score, print_on)

    return all_indexes, F1_score


def peak_classification(test_label, prediction_label, print_on=True):
    # Returns the calculate performance of the chosen classifier using the F1 score

    # Extract the labels predicted by the model
    pred_label = [x[0] for x in prediction_label]
    
    # Calculate the weight f1 score as a harmonic mean between the precision and recall, to give a better estimate of the model's performance
    weighted_f1_score = metrics.f1_score(test_label, pred_label, average="weighted")
    
    if print_on:
        # Print the score of each k fold iteration
        print("*"*20)
        print(f"Classification Weighted F1 score (%) = {round(weighted_f1_score*100, 2)}")
        print("*"*20)

    return weighted_f1_score


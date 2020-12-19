
def peak_location(incorrect_indexes, success_rate, print_on=True):
    if print_on:
        print("*"*20)
        print("Sample of Incorrect Indexes")
        i = 0
        for index in incorrect_indexes:
            if i % 20 == 0:
                print(index)
            i += 1
        
        print("*"*20)
        print("Peak Location Accuracy (%) = " + str(round(success_rate*100, 2)))
        print("*"*20)


def peak_classification(test_label, prediction_label, print_on=True):
    

    correct = sum([1 for compare in zip(test_label,prediction_label) if compare[0] == compare[1]])

    score = correct/len(prediction_label)

    if print_on:
        print("*"*20)
        print("Number correctly classified: "+ str(correct))

        print("Number of predictions: " + str(len(prediction_label)))

        print("Peak Classification Accuracy (%) = "+ str(round(score*100, 2)))
        print("*"*20)

    return score


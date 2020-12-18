
def peak_location(incorrect_indexes, success_rate):

    print("*"*20)
    print("Sample of Incorrect Indexes")
    i = 0
    for index in incorrect_indexes:
        if i % 20 == 0:
            print(index)
        i += 1
    
    print("*"*20)
    print("Peak Location Accuracy (%) = " + str(int(success_rate*100)))

    print("*"*20)


def peak_classification(test_label, prediction_label):
    

    correct = sum([1 for compare in zip(test_label,prediction_label) if compare[0] == compare[1]])

    score = correct/len(prediction_label)

    print("*"*20)
    print("Number correctly classified: "+ str(correct))

    print("Number of predictions: " + str(len(prediction_label)))

    print("Score (%) = "+ str(int(score*100)))
    print("*"*20)


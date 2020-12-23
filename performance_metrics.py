from sklearn import metrics

def peak_location(incorrect_indexes, success_rate, print_on=True):
    if print_on:
        print("*"*20)
        print("Sample of Incorrect Indexes")
        i = 0
        for index in incorrect_indexes:
            if i % 20 == 0:
                print(str(index[0]) + " - " + str(index[0]/25000))
            i += 1
        
        print("*"*20)
        print("Peak Location Accuracy (%) = " + str(round(success_rate*100, 2)))
        # print("*"*20)


def peak_classification(test_label, prediction_label, print_on=True):
    
    pred_label = [x[0] for x in prediction_label]
    # confusion = metrics.confusion_matrix(test_label, pred_label)
    # # print(confusion)

    # ind_confusion = metrics.multilabel_confusion_matrix(test_label, pred_label, labels=[1,2,3,4])
    # print(ind_confusion)

    # report = metrics.classification_report(test_label, pred_label, labels=[1,2,3,4], output_dict=True)
    # print(report)

    weighted_f1_score = metrics.f1_score(test_label, pred_label, average="weighted")#report['weighted avg']['f1-score']
    # print(weighted_f1_score)

    spike_metrics = []
    # for confuse in ind_confusion:
    #     TP = confuse[1,1]
    #     TN = confuse[0,0]
    #     FP = confuse[0,1]
    #     FN = confuse[1,0]
    #     class_acc = (TP + TN) / float(TP + TN + FP + FN)
    #     class_err = (FP + FN) / float(TP + TN + FP + FN)
    #     sensitive = TP / float(FN + TP)
    #     specific = TN / float(TN + FP)
    #     false_pos = FP / float(TN + FP)
    #     precision = TP / float(TP + FP)
    #     # recall = TP / float(TP + FN)

    #     spike = {
    #         "TP": TP,
    #         "TN": TN,
    #         "FP": FP,
    #         "FN": FN,
    #         "acc": class_acc,
    #         "err": class_err, 
    #         "sens": sensitive,
    #         "spec": specific,
    #         "false_p": false_pos,
    #         "prec": precision
    #     }
    #     spike_metrics.append(spike)

    # correct = sum([1 for compare in zip(test_label,pred_label) if compare[0] == compare[1]])

    # score = correct/len(prediction_label)

    # if print_on:
    #     print("*"*20)
    #     print("Weighted F1 score (%) = "+ str(round(weighted_f1_score*100, 2)))
    #     print("*"*20)

    return weighted_f1_score, spike_metrics


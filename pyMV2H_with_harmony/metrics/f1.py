from math import isnan


def f1_score(true_positives, false_positives, false_negatives,return_details=False):
    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2. * recall * precision / (recall + precision)
        if return_details:
            return  0. if isnan(f1) else f1, precision, recall
        return 0. if isnan(f1) else f1
    except ZeroDivisionError:
        if return_details:
            return 0., 0., 0.
        return 0.

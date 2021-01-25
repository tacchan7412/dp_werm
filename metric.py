import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import balanced_accuracy_score


def error(scores, labels):
    scores[scores > 0] = 1
    scores[scores <= 0] = -1
    return np.sum(np.abs(scores - labels) / 2) / float(np.size(labels))


def f1score(scores, labels):
    scores[scores > 0] = 1
    scores[scores <= 0] = 0
    return f1_score((labels+1)/2, scores)


def auc_pr(scores, labels):
    preds = 1 / (1 + np.exp(-scores))
    precision, recall, _ = precision_recall_curve((labels+1)/2, preds)
    return auc(recall, precision)


def auc_roc(scores, labels):
    preds = 1 / (1 + np.exp(-scores))
    return roc_auc_score((labels+1)/2, preds)


def balanced_acc(scores, labels):
    scores[scores > 0] = 1
    scores[scores <= 0] = 0
    return balanced_accuracy_score((labels+1)/2, scores)


def recall(scores, labels):
    scores[scores > 0] = 1
    scores[scores <= 0] = 0
    return recall_score((labels+1)/2, scores)


def precision(scores, labels):
    scores[scores > 0] = 1
    scores[scores <= 0] = 0
    return precision_score((labels+1)/2, scores)

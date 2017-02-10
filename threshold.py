from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import sys
sys.setrecursionlimit(5000000)


def f(index, current_label, labels, num_labels, dp_matrix, class_weight):
    if index >= len(labels) or current_label >= num_labels:
        return 0

    if dp_matrix[index][current_label] != -1:
        return dp_matrix[index][current_label]

    error = class_weight[labels[index]][current_label]

    if current_label + 1 == num_labels:
        dp_matrix[index][current_label] = \
            error + \
            f(index + 1, current_label, labels, num_labels, dp_matrix, class_weight)
    else:
        dp_matrix[index][current_label] = \
            min(error +
                f(index + 1, current_label, labels, num_labels, dp_matrix, class_weight),
                f(index, current_label + 1, labels, num_labels, dp_matrix, class_weight))
    return dp_matrix[index][current_label]


def _decide_thresholds(scores, labels, num_labels, class_weight):
    def traverse_matrix(dp_matrix, class_weight):
        nscores, nlabels = dp_matrix.shape
        index, current_label = 0, 0
        ret = []
        while index+1 < nscores and current_label+1 < num_labels:
            current = dp_matrix[index][current_label]
            keep = dp_matrix[index + 1][current_label]
            error = class_weight[labels[index]][current_label]
            if abs((current - error) - keep) < 1e-5:
                index += 1
            else:
                ret.append(index)
                current_label += 1
        return ret

    dp_matrix = -np.ones((len(labels), num_labels), dtype=np.float32)
    f(0, 0, labels, num_labels, dp_matrix, class_weight)
    path = traverse_matrix(dp_matrix, class_weight)

    #return scores[path]  # old behavior: return midpoints
    ths = np.asarray([(scores[p]+scores[max(p-1, 0)])/2 for p in path])
    return ths


def decide_thresholds(scores, y, k, strategy, full=False):
    if strategy == 'uniform':
        w = 1-np.eye(k)
    elif strategy == 'inverse':
        w = np.repeat(len(y) / (k*(np.bincount(y)+1)), k).reshape((k, k)) * (1-np.eye(k))
    elif strategy == 'absolute':
        w = [[np.abs(i-j) for i in range(k)] for j in range(k)]
    elif strategy == 'inverse-absolute':
        w1 = np.repeat(len(y) / (k*(np.bincount(y)+1)), k).reshape((k, k)) * (1-np.eye(k))
        w2 = [[np.abs(i-j) for i in range(k)] for j in range(k)]
        w = w1*w2
    else:
        raise 'No such threshold strategy: %s' % strategy
    return _decide_thresholds(scores, y, k, w)


def threshold_to_class(s, ths):
    return np.sum(s >= ths[:, np.newaxis], 0, int)


class Threshold(BaseEstimator, ClassifierMixin):
    def __init__(self, model, strategy):
        self.model = model
        self.strategy = strategy

    def fit(self, X, y, fine_y=None):
        # self.classes_ = np.unique(y)  # required by sklearn
        # for our datasets, this makes more sense:
        self.classes_ = np.arange(np.amax(y)+1, dtype=int)

        self.model.fit(X, fine_y if fine_y is not None else y)
        s = self.model.predict_proba(X)

        # this class ensure that scores are ordered
        i = np.argsort(s)
        s = s[i]
        y = y[i]

        self.ths = decide_thresholds(
            s, y, len(self.classes_), self.strategy)

        return self

    def predict(self, X):
        s = self.model.predict_proba(X)
        yp = threshold_to_class(s, self.ths)
        return yp

    def predict_proba(self, X):
        return self.model.predict_proba(X)

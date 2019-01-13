from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        N = len(X)
        y = np.zeros(N)

        # returns distances in a matrix
        # of shape (N_test, N_train)
        distances = pairwise_distances(X, self.X)
        

        # now get the minimum k elements' indexes
        # https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
        idx = distances.argsort(axis=1)[:, :self.k]

        # now determine the winning votes
        # each row of idx contains indexes from 0..Ntrain
        # corresponding to the indexes of the closest samples
        # from the training set
        # NOTE: if you don't "believe" this works, test it
        # in your console with simpler arrays
        votes = self.y[idx]

        # now y contains the classes in each row
        # e.g.
        # sample 0 --> [class0, class1, class1, class0, ...]
        # unfortunately there's no good way to vectorize this
        # https://stackoverflow.com/questions/19201972/can-numpy-bincount-work-with-2d-arrays
        for i in range(N):
            y[i] = np.bincount(votes[i]).argmax()

        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)



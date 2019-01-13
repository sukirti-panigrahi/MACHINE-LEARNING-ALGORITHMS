# data is from https://www.kaggle.com/c/digit-recognizer
# each image is a D = 28x28 = 784 dimensional vector
# there are N = 42000 samples
# you can plot an image by reshaping to (28,28) and using plt.imshow()
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input

__author__ = "kennedy Czar"
__email__ = "kennedyczar@gmail.com"
__version__ = '1.0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from kmeans import plot_k_means, get_simple_data
from datetime import datetime
# from gmm import gmm
from sklearn.mixture import GaussianMixture
from kmeans_mnist import get_data, purity, DBI


def main():
    X, Y = get_data(10000)
    print("Number of data points:", len(Y))

    model = GaussianMixture(n_components=10)
    model.fit(X)
    M = model.means_
    R = model.predict_proba(X)

    print("Purity:", purity(Y, R)) # max is 1, higher is better
    print("DBI:", DBI(X, M, R)) # lower is better


if __name__ == "__main__":
    main()

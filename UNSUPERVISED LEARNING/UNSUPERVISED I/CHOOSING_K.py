from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
from KMEANS import plot_k_means, get_simple_data, cost

__author__ = "kennedy Czar"
__email__ = "kennedyczar@gmail.com"
__version__ = '1.0'

def main():
  X = get_simple_data()

  plt.scatter(X[:,0], X[:,1])
  plt.show()

  costs = np.empty(10)
  costs[0] = None
  for k in range(1, 10):
    M, R = plot_k_means(X, k, show_plots=False)
    c = cost(X, R, M)
    costs[k] = c

  plt.plot(costs)
  plt.title("Cost vs K")
  plt.show()


if __name__ == '__main__':
  main()

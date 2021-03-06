# MACHINE-LEARNING-ALGORITHMS [![HitCount](http://hits.dwyl.io/kennedyCzar/https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS.svg)](http://hits.dwyl.io/kennedyCzar/https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS)
Machine Learning Algorithms built from scratch(Supervised, Unsupervised, Reinforcement learning algos)
----------------------------------
A collection of key Machine Learning and Fundamental Deep Learning Algorithms Available For Free Use under [![GNU license](https://img.shields.io/badge/License-GNU-blue.svg)](https://lbesson.mit-license.org/)
------------------------------------
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## Supervised Learning

Bayes [Bayesian Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/SUPERVISED%20LEARNING/BAYES.py)

    Bayes' theorem (alternatively Bayes' law or Bayes' rule) 
    describes the probability of an event, based on prior knowledge of
    conditions that might be related to the event
    
NaiveBayes [NaiveBayes Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/SUPERVISED%20LEARNING/NAIVEBAYES.py)

    # NaiveBayes: naive Bayes are a family of simple
    "probabilistic classifiers" based on applying Bayes' theorem with strong
    (naive) independence assumptions between the features.
    
Decision tree [Decision tree Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/SUPERVISED%20LEARNING/DECISION_TREE.py)

    decision tree is a decision support tool that uses a 
    tree-like model of decisions and their possible consequences,
    including chance event outcomes, resource costs, and utility.
    
Perceptron [Preceptron Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/SUPERVISED%20LEARNING/PERCEPTRON.py)

    It is a type of linear classifier, i.e. a classification 
    algorithm that makes its predictions based on a linear predictor
    function combining a set of weights with the feature vector.
    
Regression with gradient descent [Regression with gradient descent Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/SUPERVISED%20LEARNING/REGRESSION_WTH_GRADIENT_DESCENT.py)

    It is a type of linear classifier, i.e. a classification 
    algorithm that makes its predictions based on a linear predictor
    function combining a set of weights with the feature vector.
    
KNN [KNN Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/SUPERVISED%20LEARNING/KNN.py)

    In k-NN classification, the output is a class membership. 
    An object is classified by a plurality vote of its neighbors, 
    with the object being assigned to the class most common among 
    its k nearest neighbors (k is a positive integer, typically small).
    If k = 1, then the object is simply assigned to the class of that
    single nearest neighbor.
    In k-NN regression, the output is the property value for the object.
    This value is the average of the values of its k nearest neighbors.
    
Regularized L1 Regression or Lasso [Regularized L1 Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/SUPERVISED%20LEARNING/L1_REGRESSION_REGULARIZED.py)

    In L1 regression we shrink the parameters to zero. 
    When input features have weights closer to zero that leads 
    to sparse L1 norm. In Sparse solution majority of the input features
    have zero weights and very few features have non zero weights.
    
Regularized L2 Regression or Ridge [Regularized L2 Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/SUPERVISED%20LEARNING/L2_REGRESSION_REGULARIZED.py)

    In L2 regularization, regularization term is the sum of 
    square of all feature weights.
    L2 regularization forces the weights to be small but does
    not make them zero and does non sparse solution.
    
Adaboost [Adaboost Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/SUPERVISED%20LEARNING/ADABOOST.py)

    The output of the other learning algorithms ('weak learners') 
    is combined into a weighted sum that represents the final output 
    of the boosted classifier. AdaBoost is adaptive(Hence the name Adaptive Boosting) 
    in the sense that subsequent weak learners are tweaked in favor of those instances 
    misclassified by previous classifiers.
    
    
## Unsupervised Learning

KMEANS [KMEANS Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/UNSUPERVISED%20LEARNING/UNSUPERVISED%20I/KMEANS.py)

    k-means clustering aims to partition n observations 
    into k clusters in which each observation belongs to the
    cluster with the nearest mean, serving as a prototype of the cluster.
    
HIERARCHICAL CLUSTERING [HIERARCHICAL clusetring Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/UNSUPERVISED%20LEARNING/UNSUPERVISED%20I/HCLUSTER.py)

    k-means clustering aims to partition n observations 
    into k clusters in which each observation belongs to the
    cluster with the nearest mean, serving as a prototype of the cluster.
    
    
GMM [GMM Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/UNSUPERVISED%20LEARNING/UNSUPERVISED%20I/GMM.py)

    A Gaussian mixture model (GMM) is a category of 
    probabilistic model which states that all generated data points are
    derived from a mixture of a finite Gaussian distributions that has no
    known parameters. The parameters for Gaussian mixture models are
    derived either from maximum a posteriori estimation or an iterative
    expectation-maximization algorithm from a prior model which is well trained.
    Gaussian mixture models are very useful when it comes to modeling data, 
    especially data which comes from several groups
    

RBM [RBM Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/UNSUPERVISED%20LEARNING/UNSUPERVISED%20II/RBM.py)

    A restricted Boltzmann machine (RBM) is a generative 
    stochastic artificial neural network that can learn a
    probability distribution over its set of inputs.

PCA IMPLEMENTATION [PCA IMPLEMENTATION](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/UNSUPERVISED%20LEARNING/UNSUPERVISED%20II/PCA_IMPL.py)

        Implementation of sklearn PCA using kaggle MNIST dataset
        
TSNE IMPLEMENTATION [TSNE IMPLEMENTATION](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/UNSUPERVISED%20LEARNING/UNSUPERVISED%20II/TSNE_BOOK.py)

        Implementation of sklearn TSNE using characters from a book

TSNE IMPLEMENTATION FOR MNIST [TSNE IMPLEMENTATION MNIST](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/UNSUPERVISED%20LEARNING/UNSUPERVISED%20II/TSNE_MNIST.py)

        Implementation of sklearn TSNE using MNIST dataset

Autoencoder [Autoencoder Algorithm](https://github.com/kennedyCzar/MACHINE-LEARNING-ALGORITHMS/blob/master/UNSUPERVISED%20LEARNING/UNSUPERVISED%20II/autoencoder.py)

    An autoencoder is a type of artificial neural network used
    to learn efficient data codings in an unsupervised manner.
    The aim of an autoencoder is to learn a representation (encoding)
    for a set of data, typically for dimensionality reduction. Along
    with the reduction side, a reconstructing side is learnt, where the 
    autoencoder tries to generate from the reduced encoding a representation as
    close as possible to its original input, hence its name.
    

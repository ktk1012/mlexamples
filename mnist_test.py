"""
mnist_test.py - Simple MNIST image classification examples.
Use scikit-learn.
Example includes svm, decision tree, random forests, majority voting, and k-NN.
And I used simple technique known as denosing (refer function; nudge dataset).
If you need more informations about scikit-learn, see https://scikit-learn.org.
 - Tae-kyeom, Kim (kimtkyeom@gmail.com)
"""
# Numpy import
import numpy as np
# Functions for machine learning algorithms
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
# For image processing
from scipy.ndimage import convolve
# Plotting data
import matplotlib.pyplot as plt

from utils import (
    maybe_download,
    extract_data,
    extract_labels)

import time


def nudge_dataset(X, y):
    """
    This produces a dataset 8 times bigger than the original one,
    by moving the 8x8 images in X around by 8 directions
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]],

        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 1],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [1, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]]
    ]

    new_images = []
    for vectors in direction_vectors:
        new_images.append(convolve(X[0].reshape((28, 28)), vectors, mode='constant'))
    new_images.append(X[0].reshape((28, 28)))
    f, axarr = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            axarr[i, j].imshow(new_images[3 * i + j], cmap='gray')

    plt.show()

    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    print X.shape
    y = np.concatenate([y for _ in range(len(direction_vectors) + 1)], axis=0)
    print y.shape
    return X, y


# Extract data
train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

X_train = extract_data(train_data_filename, 60000, dense=True)
y_train = extract_labels(train_labels_filename, 60000, one_hot=False)
X_test = extract_data(test_data_filename, 10000, dense=True)
y_test = extract_labels(test_labels_filename, 10000, one_hot=False)


#################################################
# Test for decision tree classifier without dimensionality reduction
Tree = DecisionTreeClassifier()
Tree.fit(X_train, y_train)
print 'Without dimenstionality reduction: ', Tree.score(X_test, y_test)

# Dimensionality reduction using PCA (784 -> 64)
pca = PCA(n_components=64)
pca.fit(X_train)
X_train_reduce = pca.transform(X_train)

Tree_2 = DecisionTreeClassifier()
Tree_2.fit(X_train_reduce, y_train)

X_test_reduce = pca.transform(X_test)
print 'With dim reduction to 64: ', Tree_2.score(X_test_reduce, y_test)

# Random forests classifier with 100 trees
RF = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=5)
RF.fit(X_train, y_train)

print 'Random Forest: ', RF.score(X_test, y_test)

"""
# With nudging data & random forests
RF2 = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=4)
X_new, y_new = nudge_dataset(X_train, y_train)
RF2.fit(X_new, y_new)
X_test_new, y_test_new = nudge_dataset(X_test, y_test)
print 'Nudging: ', RF2.score(X_test_new, y_test_new)
"""

# Linear svm with dim reduction
clf = svm.LinearSVC()
clf.fit(X_train_reduce, y_train)
X_test_reduce = pca.transform(X_test)
print 'Linear Svm: ', clf.score(X_test_reduce, y_test)

# k-NN with nudge data sets
clf2 = KNeighborsClassifier(n_neighbors=7)
clf2.fit(X_train, y_train)
print 'KNN :', clf2.score(X_test, y_test)

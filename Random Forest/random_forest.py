"""
Class for building a Random Forest.
"""

from decision_tree import DecisionTree
import numpy as np

class RandomForest(object):
    """
    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of the trees.
    """
    def __init__(self, num_trees, max_tree_depth=1e10, ratio_per_tree=0.5):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.ratio_per_tree = ratio_per_tree
        self.trees = None

    def fit(self, X_train, Y_train):
        """
        :param X_train: 2-dimensional python list or numpy 2-dimensional array
        :param Y_train: 1-dimensional python list or numpy 1-dimensional array
        """
        self.trees = []
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # Build self.num_trees trees of depth self.max_tree_depth
        # with randomized data.

        for _ in range(self.num_trees):
            idx = np.arange(len(X_train))
            np.random.shuffle(idx)
            idx = idx[0:int(len(X_train)*self.ratio_per_tree)]
            X_rand = X_train[idx]
            Y_rand = Y_train[idx]

            d_tree = DecisionTree(self.max_tree_depth)
            d_tree.fit(X_rand, Y_rand)
            self.trees.append(d_tree)

    def predict(self, X_test):
        """
        :param X_test: 2 dimensional python list or numpy 2-dimensional array
        :return: (Y_pred, conf), tuple with Y_pred being 1-dimensional python
        list with labels, and conf being 1-dimensional list with
        confidences for each of the labels.
        """
        # Evaluate labels in each of the `self.tree`s and return the
        # label and confidence with the most votes for each of
        # the data points in `X_test`
        predicted = []
        Y_pred = []
        conf = []
        for tree in self.trees:
            predicted.append(list(tree.predict(X_test)))
        predicted = np.transpose(np.array(predicted))
        for i, _ in enumerate(predicted):
            prob, label = most_frequent(list(predicted[i]))
            Y_pred.append(label)
            conf.append(prob)
        return (Y_pred, conf)

def most_frequent(X_list):
    """
    :param X_list: 1-dimensional python list or numpy 1-dimensional array
    :return: tuple of the most frequent value and its relative frequency in X_list
    """
    relfreq = []
    for i in X_list:
        relfreq.append(X_list.count(i)/len(X_list))
    return max(zip(relfreq, X_list))

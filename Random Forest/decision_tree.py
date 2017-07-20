"""
Class to build a DecisionTree.
"""

import numpy as np
from decision_node import build_tree

class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """
    def __init__(self, max_tree_depth=1e10):
        self.max_tree_depth = max_tree_depth
        self.tree = None

    def fit(self, x_train, y_train):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        data = np.column_stack((x_train, y_train))
        # Build a tree that has self.max_depth
        self.tree = build_tree(data, max_depth=self.max_tree_depth)


    def predict_for_one(self, x_one, tree):
        """
        Function to assist predict(X)
        """

        if tree.is_leaf:
            tmp = list(tree.current_results)
            res = tmp[0]
            return res
        else:
            if isinstance(x_one[tree.column], (int, float)):
                if str(x_one[tree.column]) >= str(tree.value):
                    return self.predict_for_one(x_one, tree.true_branch)
                else:
                    return self.predict_for_one(x_one, tree.false_branch)
            else:
                if str(x_one[tree.column]) == str(tree.value):
                    return self.predict_for_one(x_one, tree.true_branch)
                else:
                    return self.predict_for_one(x_one, tree.false_branch)



    def predict(self, x_test):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimensional python list with labels
        """
        # Evaluate label of all the elements in `X` and
        # return same size list with labels.

        y_pred = []
        for i, _ in enumerate(x_test):
            correct_class = self.predict_for_one(x_test[i], self.tree)
            y_pred.append(correct_class)
        return y_pred

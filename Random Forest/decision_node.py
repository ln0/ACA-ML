"""
Class to build a DecisionNode.
"""

from collections import defaultdict
import numpy as np

class DecisionNode(object):
    """
    README
    DecisionNode is a building block for Decision Trees.
    DecisionNode is a python class representing a  node in our decision tree
    node = DecisionNode()  is a simple usecase for the class
    you can also initialize the class like this:
    node = DecisionNode(column = 3, value = "Car")
    In python, when you initialize a class like this, its __init__ method is called
    with the given arguments. __init__() creates a new object of the class type, and initializes its
    instance attributes/variables.
    In python the first argument of any method in a class is 'self'
    Self points to the object which it is called from and corresponds to 'this' from Java

    """

    def __init__(self,
                 column=None,
                 value=None,
                 false_branch=None,
                 true_branch=None,
                 current_results=None,
                 is_leaf=False,
                 results=None):
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.current_results = current_results
        self.is_leaf = is_leaf
        self.results = results


def dict_of_values(data):
    """
    param data: a 2D Python list representing the data. Last column of data is Y.
    return: returns a python dictionary showing how many times each value appears in Y

    for example
    data = [[1,'yes'],[1,'no'],[1,'yes'],[1,'yes']]
    dict_of_values(data)
    should return {'yes' : 3, 'no' :1}
    """
    results = defaultdict(int)
    for row in data:
        r = row[len(row) - 1]
        results[r] += 1
    return dict(results)


def divide_data(data, feature_column, feature_val):
    """
    this function takes the data and divides it in two parts by a line. A line
    is defined by the feature we are considering (feature_column) and the target
    value. The function returns a tuple (data1, data2) which are the desired parts of the data.
    For int or float types of the value, data1 have all the data with values >= feature_val
    in the corresponding column and data2 should have rest.
    For string types, data1 should have all data with values == feature val and data2 should
    have the rest.

    param data: a 2D Python list representing the data. Last column of data is Y.
    param feature_column: an integer index of the feature/column.
    param feature_val: can be int, float, or string
    return: a tuple of two 2D python lists
    """

    data1 = []
    data2 = []

    if isinstance(feature_val, (int, float)):
        for i in range(len(data)):
            if data[i][feature_column] >= feature_val:
                data1.append(data[i])
            else: data2.append(data[i])
    else:
        for i in range(len(data)):
            if data[i][feature_column] == feature_val:
                data1.append(data[i])
            else: data2.append(data[i])

    return (data1, data2)


def gini_impurity(data1, data2):

    """
    Given two 2D lists of compute their gini_impurity index.
    Remember that last column of the data lists is the Y
    Lets assume y1 is y of data1 and y2 is y of data2.
    gini_impurity shows how diverse the values in y1 and y2 are.
    gini impurity is given by

    n1*sum(p_k1 * (1-p_k1)) + n2*sum(p_k2 * (1-p_k2))

    where N1 is number of points in data1
    p_k1 is fraction of points that have y value of k in data1
    same for N2 and p_k2


    param data1: A 2D python list
    param data2: A 2D python list
    return: a number - gini_impurity
    """

    n1 = len(data1)
    n2 = len(data2)

    p_k1 = np.array(list(dict_of_values(data1).values())) / n1
    p_k2 = np.array(list(dict_of_values(data2).values())) / n2

    return n1 * np.sum(p_k1 * (1 - p_k1)) + n2 * np.sum(p_k2 * (1 - p_k2))



def build_tree(data, current_depth=0, max_depth=1e10):
    """
    build_tree is a recursive function.
    What it does in the general case is:
    1: find the best feature and value of the feature to divide the data into
    two parts
    2: divide data into two parts with best feature, say data1 and data2
        recursively call build_tree on data1 and data2. this should give as two
        trees say t1 and t2. Then the resulting tree should be
        DecisionNode(...... true_branch=t1, false_branch=t2)


    In case all the points in the data have same Y we should stop splitting and return that node
    For this function we will give you some of the code so its not too hard for you ;)

    param data: param data: A 2D python list
    param current_depth: an integer. This is used if we want to limit the numbr of layers in the
        tree
    param max_depth: an integer - the maximal depth of the representing
    return: an object of class DecisionNode

    """

    if len(data) == 0:
        return DecisionNode(is_leaf=True)

    elif current_depth == max_depth:
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)

    elif len(dict_of_values(data)) == 1:
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)

    #This calculates gini number for the data before dividing
    self_gini = gini_impurity(data, [])

    #Below are the attributes of the best division that you need to find.
    #You need to update these when you find a division which is better

    best_gini = 1e100
    best_column = None
    best_value = None
    #best_split is tuple (data1,data2) which shows the two datas for the best divison so far
    best_split = None

    # You need to find the best feature to divide the data
    # For each feature and each possible value of the feature compute the
    # gini number for that division. You need to find the feature that minimizes
    # gini number. Remember that last column of data is Y
    # Think how you can use the divide_data and gini_impurity functions you wrote
    # above
    for col in range(len(data[0]) - 1):
        k = list(set(row[col] for row in data))
        for val in k:
            data1, data2 = divide_data(data, col, val)
            gini = gini_impurity(data1, data2)
            if gini < best_gini:
                best_gini = gini
                best_column = col
                best_value = val
                best_split = (data1, data2)

    #if best_gini is no improvement from self_gini, we stop and return a node.
    if abs(self_gini - best_gini) < 1e-10:
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)
    else:
        t_branch = build_tree(best_split[0], current_depth + 1, max_depth - 1)
        f_branch = build_tree(best_split[1], current_depth + 1, max_depth - 1)

    return DecisionNode(true_branch=t_branch, false_branch=f_branch, value=best_value, column=best_column, current_results=dict_of_values(data), is_leaf=False)


def print_tree(tree, indent=''):
    """
    Prints the tree in a readable format.
    """

    # Is this a leaf node?
    if tree.is_leaf:
        print(str(tree.current_results))
    else:
        # Print the criteria
        print(indent+'Current Results: ' + str(tree.current_results))
        print('Column ' + str(tree.column) + ' : ' + str(tree.value) + '? ')

        # Print the branches
        print(indent + 'True->', end="")
        print_tree(tree.true_branch, indent + '  ')
        print(indent + 'False->', end="")
        print_tree(tree.false_branch, indent + '  ')



def main():
    """
    Test build_tree on data.
    """
    data = [['slashdot', 'USA', 'yes', 18, 'None'],
            ['google', 'France', 'yes', 23, 'Premium'],
            ['reddit', 'USA', 'yes', 24, 'Basic'],
            ['kiwitobes', 'France', 'yes', 23, 'Basic'],
            ['google', 'UK', 'no', 21, 'Premium'],
            ['(direct)', 'New Zealand', 'no', 12, 'None'],
            ['(direct)', 'UK', 'no', 21, 'Basic'],
            ['google', 'USA', 'no', 24, 'Premium'],
            ['slashdot', 'France', 'yes', 19, 'None'],
            ['reddit', 'USA', 'no', 18, 'None'],
            ['google', 'UK', 'no', 18, 'None'],
            ['kiwitobes', 'UK', 'no', 19, 'None'],
            ['reddit', 'New Zealand', 'yes', 12, 'Basic'],
            ['slashdot', 'UK', 'no', 21, 'None'],
            ['google', 'UK', 'yes', 18, 'Basic'],
            ['kiwitobes', 'France', 'yes', 19, 'Basic']]

    tree = build_tree(data)
    print_tree(tree)


if __name__ == '__main__':
    main()

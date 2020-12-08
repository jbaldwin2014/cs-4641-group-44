import numpy as np
from collections import Counter
from scipy import stats
from util import *

class MyDecisionTree(object):
    def __init__(self, max_depth=3):
        """
        Helper Function:
        Initializing the tree as an empty dictionary.

        Args:
        max_depth: maximum depth of the tree including the root node.
        Please consider the root node as being in depth = 0.
        """

        self.max_depth = max_depth
        self.tree = {
            'isLeaf': False,
            'split_attribute': -1,
            'split_value': '',
            'is_categorical': False,
            'leftTree': None,
            'rightTree': None,
            'depth': 0
        };


    def fit(self, X, y, depth):
        """
        TODO:        [15 points]

        Train the decision tree (self.tree) using the the sample X and labels y.

        NOTE: You will have to make use of the utility functions to train the tree.
        One possible way of implementing the tree: Each node in self.tree could be in the form of a dictionary:
        https://docs.python.org/2/library/stdtypes.html#mapping-types-dict

        For example, a non-leaf node with two children can have a 'left' key and  a  'right' key.
        You can add more keys which might help in classification (eg. split attribute and split value)


        While fitting a tree to the data, you will need to check to see if the node is a leaf node(
        based on the stopping condition explained above) or not.
        If it is not a leaf node, find the best feature and attribute split:
        X_left, X_right, y_left, y_right, for the data to build the left and
        the right subtrees.

        Remember for building the left subtree, pass only X_left and y_left and for the right subtree,
        pass only X_right and y_right.

        Args:

        X: N*D matrix corresponding to the data points
        Y: N*1 array corresponding to the labels of the data points
        depth: depth of node of the tree

        """

        # Hint: the structure of fit function is provided for you here. You have to
        # implement the buildTree function by yourself to recursively build and add nodes to decision tree.

        X = X.astype('object')

        self.tree = self.buildTree(X, y, 0)
        self.tree = self.tree.tree


    def buildTree(self, X, y, depth):
        """
            Recursively build and add nodes

        """
        # check if we need to stop splitting

        # find best feature and attribute

        if (y.size == 0):
            y = [0]
        if (np.unique(y).size == 1):
            val = stats.mode(y, axis=None)[0][0]
            node = MyDecisionTree(self.max_depth - depth)
            node.tree = {
                'feature': -1,
                'value': val,
                'IG': 0,
                'feature_num': -1,
                'isLeaf': True,
                'is_categorical': False,
                'leftTree': None,
                'rightTree': None,
                'depth': depth
            }
            return node
        if (depth == self.max_depth):
            val = stats.mode(y, axis=None)[0][0]
            node = MyDecisionTree(self.max_depth - depth)
            node.tree = {
                'feature': -1,
                'value': val,
                'IG': 0,
                'feature_num': -1,
                'isLeaf': True,
                'is_categorical': False,
                'leftTree': None,
                'rightTree': None,
                'depth': depth
            }
            return node
        feature, index, IG, SplitVal = find_best_feature(X, y)
        X_left, X_right, y_left, y_right = partition_classes(X, y, index, SplitVal)
        leftTree = self.buildTree(X_left, y_left, depth + 1)  # Be careful what should be the depth here
        rightTree = self.buildTree(X_right, y_right, depth + 1)
        node = MyDecisionTree(self.max_depth - depth)   # Be careful what should be the depth here
        node.tree = {
            'feature': feature,
            'value': SplitVal,
            'IG': IG,
            'feature_num': index,
            'isLeaf': False,
            'is_categorical': False,
            'leftTree': leftTree,
            'rightTree': rightTree,
            'depth': depth
        }
        return node


    def predict(self, record):
        """
        TODO:        [10 points]

        classify a sample in test data set using self.tree and return the predicted label

        Args:

        record: D*1, a single data point that should be classified

        Returns: True if the predicted class label is 1, False otherwise


        """

        curr = self
        while True:
            curr = curr.tree
            if curr['isLeaf']:
                if curr['value'] == 1:
                    return True
                else:
                    return False
            value = curr['feature_num']
            test = record[value]
            if (type(test) == str):
                if (record[curr['feature_num']] == curr['value']):
                    curr = curr['leftTree']
                else:
                    curr = curr['rightTree']
            else:
                if (record[curr['feature_num']] <= curr['value']):
                    curr = curr['leftTree']
                else:
                    curr = curr['rightTree']

    def DecisionTreeEvalution(self,X,y, verbose=False):
        # helper function. You don't have to modify it
        # Make predictions
        # For each test sample X, use our fitting dt classifer to predict
        y_predicted = []
        for record in X:
            y_predicted.append(self.predict(record))

        # Comparing predicted and true labels
        results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        if verbose:
            print("accuracy: %.4f" % accuracy)
        return accuracy

    def DecisionTreeError(self, y):
        # helper function for calculating the error of the entire subtree if converted to a leaf with majority class label.
        # You don't have to modify it
        num_ones = np.sum(y)
        num_zeros = len(y) - num_ones
        return 1.0 - max(num_ones, num_zeros) / float(len(y))
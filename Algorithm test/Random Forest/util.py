import numpy as np
from math import log2, sqrt


def entropy(class_y):
    """
    Input:
        - class_y: list of class labels (0's and 1's)
    Output:
        - entropy: a scalar, the value of entropy.
    TODO:     [3 points]

    Compute the entropy for a list of classes
    Example: entropy([0,0,0,1,1,1,1,1]) = 0.9544
    """
    x = np.mean(class_y)
    if x == 0 or x == 1:
        return 0
    else:
        return - x * np.log2(x) - (1 - x) * np.log2(1 - x)

def information_gain(previous_y, current_y):
    """
    Inputs:
        - previous_y : the distribution of original labels (0's and 1's)
        - current_y  : the distribution of labels after splitting based on a particular
                     split attribute and split value
    Output:
        - information_gain: a scalar, the value of information_gain.

    TODO:     [3 points]

    Compute and return the information gain from partitioning the previous_y labels into the current_y labels.

    Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf

    Example: previous_y = [0,0,0,1,1,1], current_y = [[0,0], [1,1,1,0]], info_gain = 0.4591
    """
    x = len(current_y[0]) / len(previous_y)
    if x == 0 or x == 1:
        return 0
    else:
        return entropy(previous_y) - (x * entropy(current_y[0]) + (1 - x) * entropy(current_y[1]))

def partition_classes(X, y, split_attribute, split_val):
    """
    Inputs:
    - X               : (N,D) list containing all data attributes
    - y               : a list of labels
    - split_attribute : column index of the attribute to split on
    - split_val       : either a numerical or categorical value to divide the split_attribute

    Outputs:
        - X_left, X_right, y_left, y_right : see the example below.

    TODO:    [3 points]

    Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.

    Example:

    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]

    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.

    Consider the case where we call the function with split_attribute = 0 (the index of attribute) and split_val = 3 (the value of attribute).
    Then we divide X into two lists - X_left, where column 0 is <= 3 and X_right, where column 0 is > 3.

    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]

    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.

    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]

    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]


    Return in this order: X_left, X_right, y_left, y_right
    """

    X = np.array(X, dtype=object)
    y = np.array(y)

    #######################################################################################################
    # Both list and numpy arrays are allowed in util functions. However, the dataset in the parts below is#
    # imported as numpy array. Therefore, we strongly recommend implementing as numpy array to make sure  #
    # the autograder is stable. It will also reduce the run time for decision tree and random forest.     #
    # So please keep the lines above.                                                                     #
    #######################################################################################################

    X_left = np.copy(X)
    X_right = np.copy(X)
    if type(split_val) == str:
        rightArr = X[:, [split_attribute]]
        leftArr = X[:, [split_attribute]]
        a = np.where(rightArr != split_val)[0]
        b = np.where(leftArr == split_val)[0]
        X_left = X_left[b, :]
        X_right = X_right[a, :]
        y_left = y[b]
        y_right = y[a]
    else:
        rightArr = X[:, [split_attribute]]
        leftArr = X[:, [split_attribute]]
        a = np.where(rightArr > split_val)[0]
        b = np.where(leftArr <= split_val)[0]
        X_left = X_left[b, :]
        X_right = X_right[a, :]
        y_left = y[b]
        y_right = y[a]
    return X_left, X_right, y_left, y_right

def find_best_split(X, y, split_attribute):
    """
    Inputs:
        - X               : (N,D) list containing all data attributes
        - y               : a list array of labels
        - split_attribute : Column of X on which to split
    Outputs:
        - best_split_val, info_gain : see the example below.

    TODO:    [3 points]

    Compute and return the optimal split value for a given attribute, along with the corresponding information gain

    Note: You will need the functions information_gain and partition_classes to write this function.
    It is recommended that when dealing with numerical values, instead of discretizing the variable space, that you loop over the unique values in your dataset
    (Hint: np.unique is your friend)

    Example:

        X = [[3, 'aa', 10],                 y = [1,
             [1, 'bb', 22],                      1,
             [2, 'cc', 28],                      0,
             [5, 'bb', 32],                      0,
             [4, 'cc', 32]]                      1]

        split_attribute = 0

        Starting entropy: 0.971

        Calculate information gain at splits:
           split_val = 1  -->  info_gain = 0.17
           split_val = 2  -->  info_gain = 0.01997
           split_val = 3  -->  info_gain = 0.01997
           split_val = 4  -->  info_gain = 0.32
           split_val = 5  -->  info_gain = 0

       best_split_val = 4; info_gain = .32;
    """
    X = np.array(X, dtype = object)

    info_gain = -1
    best_split_val = None
    tried = []
    for i in range(len(X)):
        SplitVal = X[i][split_attribute]
        mask = ~(np.isin(SplitVal, tried))
        if mask:
            X_left, X_right, y_left, y_right = partition_classes(X, y, split_attribute, SplitVal)
            tried.append(SplitVal)
        IG = information_gain(y, [y_left, y_right])
        if not np.isnan(IG) and IG > info_gain:
            info_gain = IG
            best_split_val = SplitVal
    return best_split_val, info_gain


def find_best_feature(X, y):
    """
    Inputs:
        - X: (N,D) list containing all data attributes
        - y : a list of labels

    Outputs:
        - best_split_feature, best_split_val: see the example below.

    TODO:    [3 points]

    Compute and return the optimal attribute to split on and optimal splitting value

    Note: If two features tie, choose one of them at random

    Example:

        X = [[3, 'aa', 10],                 y = [1,
             [1, 'bb', 22],                      1,
             [2, 'cc', 28],                      0,
             [5, 'bb', 32],                      0,
             [4, 'cc', 32]]                      1]

        split_attribute = 0

        Starting entropy: 0.971

        Calculate information gain at splits:
           feature 0:  -->  info_gain = 0.32
           feature 1:  -->  info_gain = 0.17
           feature 2:  -->  info_gain = 0.4199

       best_split_feature: 2 best_split_val: 22
    """
    X = np.array(X, dtype = object)

    temp = -1
    myData = ["Lab-Confirmed Case","Male","Age","Race","Hospitalized","ICU Patient","Pre-existing"]
    for i in range(len(X[0])):
        temp_val, IG = find_best_split(X, y, i)
        if not np.isnan(IG) and IG > temp:
            temp = IG
            index = i
            infoGain = IG
            best_split_feature = myData[i]
            best_split_val = temp_val
    return best_split_feature, index, infoGain, best_split_val

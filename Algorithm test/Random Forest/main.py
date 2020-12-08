import numpy as np
import decision_tree
import random_forest
from collections import Counter
from scipy import stats
from math import log2, sqrt
import pandas as pd
import time
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_circles
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

def recursive_printing(root, string):
    if (root["isLeaf"]):
        if (root["depth"] == 4):
            if (root["value"] == 0):
                string = string + "{\"Live\"},"
            else:
                string = string + "{\"Die\"},"
        else:
            if (root["value"] == 0):
                string = string + "{\"Live STOPPED AT DEPTH" + str(root["depth"]) + "\"},"
            else:
                string = string + "{\"Die STOPPED AT DEPTH" + str(root["depth"]) + "\"},"
        return string
    else:
        infogain = "{:.4f}".format(root["IG"])
        if (root["feature"] == "Age") or (root["feature"] == "Race"):
            string = string + "{sprintf(\"" + root["feature"] + " " + str(root["value"]) + "\\nIG: " + infogain + "\")},"
        else:
            string = string + "{sprintf(\"" + root["feature"] + "\\nIG: " + infogain + "\")},"
        #print("Depth:",root["depth"],"{Split:",root["feature"],"} {Value:",root["value"],"} {IG:",root["IG"],"}")
        string = recursive_printing(root["leftTree"].tree, string)
        string = recursive_printing(root["rightTree"].tree, string)
        return string

def print_tree(forest):
    string = ""
    for i in range(len(forest.decision_trees)):
        print("\nPrinting Tree",i+1,"of",len(forest.decision_trees),"(Preorder Traverse)\n")
        labels = recursive_printing(forest.decision_trees[i].tree, string)
        print(labels)


def main(x, size):
    # helper function. You don't have to modify it
    data = pd.read_csv("CleanData.csv")
    data = data.drop([data.columns[0]], axis='columns')
    """
    lastRow = data.shape[0]
    data_test = data.iloc[list(range(0, int(data.shape[0] / 3), 1))]
    data_valid = data.iloc[list(range(int(data.shape[0] / 3) + 1, (int(data.shape[0] / 3)) * 2, 1))]
    data_train = data.iloc[list(range(int(data.shape[0] / 3) * 2 + 1, lastRow, 1))]
    """
    testSize = int(size / 4)
    data_train = data.iloc[list(range(0, size, 1))]
    data_test = data.iloc[list(range((size + 1), (size + 1) + testSize, 1))]
    data_valid = data.iloc[list(range((size + 1) + testSize + 1, ((size + 1) + 1 + (2 * (testSize))), 1))]


    numerical = ['sex','age_group','race_ethnicity_combo','hosp_yn','icu_yn','death_yn','medcond_yn']

    X_train = data_train.drop(columns = 'death_yn')
    y_train = data_train['death_yn']
    X_test = data_test.drop(columns = 'death_yn')
    y_test = data_test['death_yn']
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    X_valid = data_valid.drop(columns = 'death_yn')
    y_valid = data_valid['death_yn']
    X_valid, y_valid = np.array(X_valid), np.array(y_valid)
    if x == 0:
        ### Initializing a decision tree.
        dt = decision_tree.MyDecisionTree(max_depth=4)

        # Building a tree
        print("Fitting the decision tree")
        dt.fit(X_train, y_train, 0)

        # Evaluating the decision tree
        dt.DecisionTreeEvalution(X_test,y_test, True)
    else:
        n_estimators = 8
        max_depth = 4
        max_features = 0.9

        myforest = random_forest.RandomForest(n_estimators, max_depth, max_features)

        myforest.fit(X_train, y_train)

        accuracy=myforest.OOB_score(X_train, y_train)

        print("accuracy: %.4f" % accuracy)
        print_tree(myforest)


if __name__ == "__main__":
    # First Arg:    0 for Decision Tree, 1 for Random Forest
    #
    # Second Arg:   Data Segment size for Estimator Training
    #               Test and Validation are each 25% size of training set
    #               i.e. 100 would mean 100 data points for training, 25 for test, and 25 for validation
    main(1, 30000)




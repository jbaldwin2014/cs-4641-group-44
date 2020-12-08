import numpy as np
from decision_tree import MyDecisionTree

"""
NOTE: You are required to use your own decision tree MyDecisionTree() to finish random forest.
"""

class RandomForest(object):
    def __init__(self, n_estimators=10, max_depth=3, max_features=0.9):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = 0.9
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [MyDecisionTree(max_depth=max_depth) for i in range(n_estimators)]

    def _bootstrapping(self, num_training, num_features, random_seed = None):
        """
        TODO: [5 pts]
        - Randomly select a sample dataset of size num_training with replacement from the original dataset.
        - Randomly select certain number of features (num_features denotes the total number of features in X,
          max_features denotes the percentage of features that are used to fit each decision tree) without replacement from the total number of features.

        Return:
        - row_idx: the row indices corresponding to the row locations of the selected samples in the original dataset.
        - col_idx: the column indices corresponding to the column locations of the selected features in the original feature list.

        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        """
        # You must set the random seed to pass the bootstrapping unit test, which is already implemented for you.
        # Since random_seed is set to None, this function will always be random except for during the bootstrapping unit test
        np.random.seed(seed = random_seed)

        return np.random.choice(num_training, num_training, replace=True), np.random.choice(num_features, int(num_features*self.max_features), replace=False)


    def bootstrapping(self, num_training, num_features):
        # helper function. You don't have to modify it
        # Initializing the bootstap datasets for each tree
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        TODO:
        Train decision trees using the bootstrapped datasets.
        Note that you need to use the row indices and column indices.

        Inputs:
        X: a matrix with num_training rows and num_features columns where num_training is the number of total records and num_features is the number of features of each record.
        y: a vector of labels of length num_training.
        """
        self.bootstrapping(len(X), len(X[0]))
        for i in range(self.n_estimators):
            print("Fitting Tree",i+1,"of",self.n_estimators,"with maximum depth",self.max_depth)
            self.decision_trees[i].fit(X[self.bootstraps_row_indices[i]][:, self.feature_indices[i]], y[self.bootstraps_row_indices[i]], 0)


    def OOB_score(self, X, y):
        # helper function. You don't have to modify it
        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(self.decision_trees[t].predict(X[i][self.feature_indices[t]]))
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)

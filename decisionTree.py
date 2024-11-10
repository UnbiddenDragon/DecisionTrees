import numpy as np
from collections import Counter

def entropy(y):
    counts = np.bincount(y)
    percentages = counts / len(y)
    entropy = 0

    for p in percentages:
        if p > 0:
            entropy += p*np.log2(p)
        return -entropy

class Node:
    """
    Helper class holding node informatin.

    Arguments:
        feature -- which column the threshold will refer to
        threshold -- the splitting parameter
        left -- left child node
        right -- right child node
        value -- label of the node, exist only if it is a leaf node
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        # if not specified the tree will use all the features
        if self.n_feats is not None:
            self.n_feats = min(self.n_feats, X.shape[1])
        else:
            self.n_feats = X.shape[1]
        self.n_features = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # pick a random subset of the features if n_feats is smaller than n_features (this is used by Random Forests)
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right, y)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            candidate_thesholds = np.unique(X_column)
            for threshold in candidate_thesholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, split_threshold):
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_threshold)

        # if all elements fall on one side or the other the split has no effect
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted average of the entropy for the children
        n = len(y)
        left_child_entropy = entropy(y[left_idxs])
        right_child_entropy = entropy(y[right_idxs])
        left_weight = len(left_idxs) / n
        right_weight = len(right_idxs) / n
        children_entropy = left_weight * left_child_entropy + right_weight * right_child_entropy

        # information gain as the difference in entropy before and after split
        information_gain = parent_entropy - children_entropy
        return information_gain

    def _split(self, X_column, split_threshold):
        # returns 2 arrays, one with all the indices of the values smaller or equal than the threshold
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        """
        Recursively traverse the tree
        """
        # if is a lead node returns its value
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
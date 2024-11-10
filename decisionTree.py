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
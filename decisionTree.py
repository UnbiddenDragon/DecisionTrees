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


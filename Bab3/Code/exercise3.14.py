# Exercise 3.14 Add Two More Points to Each Group for Label Spreading
import numpy as np
from sklearn.semi_supervised import LabelSpreading


X = np.array([
    [0, 1], [1, 1], [2, 0], [3, 1], [4, 1], [5, 0],
    [10, 5], [11, 6], [12, 4], [13, 5], [14, 6], [15, 4],
])
labels = np.full(12, -1.0)
labels[0] = 0
labels[-1] = 1
print(labels)

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)
print(label_spread.transduction_)

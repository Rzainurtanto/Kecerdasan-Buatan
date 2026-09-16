# Exercise 3.15 Add a Third Group of Points for Label Spreading
import numpy as np
from sklearn.semi_supervised import LabelSpreading


X = np.array([
    [0, 1], [1, 1], [2, 0], [3, 1],
    [10, 5], [11, 6], [12, 4], [13, 5],
    [20, 0], [21, 1], [22, -1], [23, 0],
])
labels = np.full(12, -1.0)
labels[0] = 0
labels[4] = 1
labels[8] = 2
print(labels)

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)
print(label_spread.transduction_)

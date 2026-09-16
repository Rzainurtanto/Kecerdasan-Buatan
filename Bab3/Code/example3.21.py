# Example 3.21 Semi-supervised Learning
import numpy as np
from sklearn.semi_supervised import LabelSpreading


X = np.array([[0, 1], [1, 1], [2, 0], [3, 1], [10, 5], [11, 6], [12, 4], [13, 5]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
labels = np.full(8, -1.0)
labels[0] = 0
labels[-1] = 1
print(labels)

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)
output_labels = label_spread.transduction_
print(output_labels)

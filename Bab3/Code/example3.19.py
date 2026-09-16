# Example 3.19 Logistic Regression
import numpy as np
from sklearn.linear_model import LogisticRegression


X = np.array([[0], [1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 30, 32, 31])
clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.predict([[6]]))
print(clf.predict_proba([[6]]))
print(clf.score(X, y))

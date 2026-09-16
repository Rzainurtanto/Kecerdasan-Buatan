# Exercise 3.9 Random Forest on Diabetes Data
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


X, y = load_diabetes(return_X_y=True)
y_class = (y > np.median(y)).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.5, random_state=0)
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Total points: %d Correctly labeled points : %d" % (y_test.shape[0], (y_test == y_pred).sum()))

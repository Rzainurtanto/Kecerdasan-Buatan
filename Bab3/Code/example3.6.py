# Example 3.6 Python SVM Breast Cancer Classifications
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)
clf = SVC()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1, 0]))
confusion = pd.DataFrame(
    cm,
    index=["is_cancer", "is_healthy"],
    columns=["predicted_cancer", "predicted_healthy"],
)
print(confusion)
print(classification_report(y_test, y_predict))

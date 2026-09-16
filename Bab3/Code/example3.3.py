# Example 3.3 Python SVM Iris CSV Classifications
from sklearn import svm, datasets
import pandas as pd

df = pd.read_csv('iris.csv')

X = df.values[:, :2]

s = df['species']

d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])

y = [d[x] for x in s]

clf = svm.SVC()
clf.fit(X, y)

# Predict the flower for a given Sepal length and width
p = clf.predict([[5.4, 3.2]])
print(p)
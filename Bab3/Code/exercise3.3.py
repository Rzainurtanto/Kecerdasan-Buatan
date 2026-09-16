# Exercise 3.3 Python SVM Iris URL Scatter Plot
from sklearn import svm
import pandas as pd
from matplotlib import pyplot


df = pd.read_csv(
    'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
)

pyplot.scatter(df['sepal_length'], df['sepal_width'], c='tab:blue')
pyplot.xlabel('Sepal length')
pyplot.ylabel('Sepal width')
pyplot.title('Iris Sepal Scatter Plot')
pyplot.show()

X = df.values[:, :2]
species = df['species']
labels = dict((name, index) for index, name in enumerate(sorted(set(species))))
y = [labels[name] for name in species]

clf = svm.SVC()
clf.fit(X, y)
p = clf.predict([[5.4, 3.2]])
print(p)

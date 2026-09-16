# Example 3.4 Python SVM Iris URL Classifications
from sklearn import svm
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix


df = pd.read_csv(
    "https://gist.githubusercontent.com/curran/"
    "a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/"
    "iris.csv"
)
print(df.shape)
print(df.head(10))
print(df.tail(10))
print(df.describe())

print(df.isna().sum().sum())
df = df.dropna()
print(df.groupby("species").size())

df.hist()
pyplot.show()

scatter_matrix(df)
pyplot.show()

X = df.values[:, :2]
species = df["species"]
labels = dict((name, index) for index, name in enumerate(sorted(set(species))))
y = [labels[name] for name in species]

clf = svm.SVC()
clf.fit(X, y)
p = clf.predict([[5.4, 3.2]])
print(p)

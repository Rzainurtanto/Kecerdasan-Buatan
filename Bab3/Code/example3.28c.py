# Example 3.28c The LazyPredict.ipynb Program (Part 3)
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1)

clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

plt.figure(figsize=(10, 5))
plt.plot(models.index, models['Accuracy'])
plt.xticks(rotation=90)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('LazyPredict Model Accuracy')
plt.tight_layout()
plt.show()

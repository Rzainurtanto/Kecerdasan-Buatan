# Example 3.28e The LazyPredict.ipynb Program (Part 5)
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X = X.head(5000).copy()
y = y.head(5000).copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
reg = LazyRegressor(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

plt.figure(figsize=(10, 5))
plt.plot(models.index, models['R-Squared'], '-s')
plt.xticks(rotation=90)
plt.ylabel('R-Squared')
plt.xlabel('Model')
plt.title('LazyPredict Regression R-Squared')
plt.tight_layout()
plt.show()

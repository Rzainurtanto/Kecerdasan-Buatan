# Example 3.28d The LazyPredict.ipynb Program (Part 4)
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X = X.head(5000).copy()
y = y.head(5000).copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
reg = LazyRegressor(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models.head())

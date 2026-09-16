# Exercise 3.18 Regression on California Housing Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy().rename(columns={'MedHouseVal': 'target'})

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f'R2 score: {r2_score(y_test, pred):.4f}')

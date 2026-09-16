# Exercise 3.12 Multiple Linear Regression on Linnerud Dataset
from sklearn.datasets import load_linnerud
from sklearn import linear_model
import numpy as np


linnerud = load_linnerud()
X = linnerud.data
y = linnerud.target[:, 0]
reg = linear_model.LinearRegression()
reg.fit(X, y)
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)
pred = reg.predict([X[0]])
print('Predition: \n', pred)

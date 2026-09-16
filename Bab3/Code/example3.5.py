# Example 3.5 Breast Cancer Data
from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()
print(cancer["feature_names"])
print(cancer["data"])
print(cancer.target_names)

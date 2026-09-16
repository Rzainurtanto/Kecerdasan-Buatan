# Exercise 3.7 PCA on Breast Cancer Data
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA


cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.title('Original Breast Cancer Data')
plt.show()

pca = PCA(n_components=2)
X1 = pca.fit_transform(X)

plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Data')
plt.show()

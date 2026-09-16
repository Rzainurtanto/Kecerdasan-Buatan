# Exercise 3.13 K-means with make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.0, random_state=0)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.predict([[0, 0]]))

# Example 3.20 K-means Clustering
from sklearn.cluster import KMeans
import numpy as np


X = np.array([[1, 2, 3], [1, 4, 2], [1, 0, 3], [10, 2, 4], [9, 4, 3], [11, 0, 2]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.predict([[12, 3, 1]]))

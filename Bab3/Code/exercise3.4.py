# Exercise 3.4 Breast Cancer Histogram Plot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

selected = df[['mean radius', 'mean area', 'mean texture', 'mean smoothness']]
selected.hist(bins=20)
plt.suptitle('Breast Cancer Feature Histograms')
plt.show()

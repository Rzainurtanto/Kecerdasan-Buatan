# Exercise 3.19 PyCaret Breast Cancer Classification
import pandas as pd
from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer(as_frame=True)
df = cancer.frame.copy()
df['Target'] = cancer.target
from pycaret import classification

classification.setup(data=df, target='Target')
classification.compare_models()

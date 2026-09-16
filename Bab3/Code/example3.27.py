# Example 3.27 The PyCaret_demo.py
#!pip install pycaret
import pandas as pd
from sklearn import datasets


iris = datasets.load_iris(as_frame=True)
iris.data['Target'] = iris.target
iris = iris.data
iris.head()

from pycaret import classification

classification.setup(data=iris, target='Target')
classification.compare_models()

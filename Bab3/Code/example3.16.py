# Example 3.16 Polynomial Regression
import matplotlib.pyplot as plt
import numpy as np


x = [0, 1, 2, 3, 4, 5]
y = [3, 8, 6, 6, 7, 3]
mymodel = np.poly1d(np.polyfit(x, y, 3))
print(mymodel)
myline = np.linspace(0, 5, 100)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

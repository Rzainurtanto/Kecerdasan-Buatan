# Exercise 3.11 Linear Regression with More Points and Plot Decorations
import matplotlib.pyplot as plt
from scipy import stats


x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [3, 5, 5, 6, 7, 7, 8, 9, 10, 12]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(value):
    return slope * value + intercept

mymodel = list(map(myfunc, x))
plt.scatter(x, y, label='Data points')
plt.plot(x, mymodel, color='red', label='Best fit line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Example')
plt.legend()
plt.grid(True)
plt.show()

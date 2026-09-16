import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

y1 = 3 * x + 4
y2 = 2 * x**2 + 1
y3 = x**3 + 9

plt.plot(x, y1, color="red", label="y = 3x + 4")
plt.plot(x, y2, color="blue", label="y = 2x^2 + 1")
plt.plot(x, y3, color="green", label="y = x^3 + 9")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Grafik Beberapa Fungsi Matematika")
plt.legend()
plt.grid(True)
plt.show()
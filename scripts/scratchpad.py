import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 3 * x**2 - 4 * x + 5


xs = np.arange(-5, 5, 0.25)
ys = f(xs)

plt.plot(xs, ys)
plt.show()

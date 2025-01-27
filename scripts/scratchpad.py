import matplotlib.pyplot as plt
import numpy as np
from src import Value, visualize_graph

# def f(x):
#     return 3 * x**2 - 4 * x + 5


# xs = np.arange(-5, 5, 0.25)
# ys = f(xs)

# plt.plot(xs, ys)
# plt.show()



a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

d = a + b
d.label = "d"

e = d + Value(3, label="f")
e.label = "e"

f = e * c
f.label = "f"

g = a + f
g.label = "g"

visualize_graph(g, filename="viz1")
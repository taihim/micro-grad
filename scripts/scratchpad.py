import matplotlib.pyplot as plt
import numpy as np
from src import Value, visualize_graph

# def f(x):
#     return 3 * x**2 - 4 * x + 5


# xs = np.arange(-5, 5, 0.25)
# ys = f(xs)

# plt.plot(xs, ys)
# plt.show()



h = 0.01

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b; e.label = "e"
d = e + c; d.label = "d"
f = Value(-2, label="f")
L = d * f; L.label = "L"
L.grad = 1.0

L._backward()
f._backward()
d._backward()
e._backward()
c._backward()
a._backward()
b._backward()



visualize_graph(L, filename="viz1")
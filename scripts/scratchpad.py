import numpy as np
from src import Value, visualize_graph
from src.neuron import Neuron, Layer, MLP


# a = Value(2.0, label="a")
# b = Value(-3.0, label="b")
# c = Value(10.0, label="c")
# e = a * b; e.label = "e"
# d = e + c; d.label = "d"
# f = Value(-2, label="f")
# l = d * f; l.label = "l"
# M = a * l; M.label = "M"

# M.backward()

# visualize_graph(M, filename="viz4")


# simple 2-input neuron visualization
# x1 = Value(2.0, label="x1") 
# x2 = Value(0, label="x2")

# # weights for inputs
# w1 = Value(-3.0, label="w1")
# w2 = Value(1.0, label="w2")

# # bias 
# b = Value(6.8813735870195432, label="b")

# x1w1 = x1 * w1; x1w1.label = "x1*w1"
# x2w2 = x2 * w2; x2w2.label = "x2*w2"

# x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"
# n = x1w1x2w2 + b; n.label="n"
# o = n.tanh(); o.label="o"

# o.backward()

# visualize_graph(o, filename="neuron")


# repeated value backprop. to check for a bug. add these cases as unit tests
# a = Value(3.0, label="a") # grad should be 2 because db/da of a + a = 1 + 1 => 2
# b = a + a; b.label = "b"
# b.backward()
# visualize_graph(b, filename="repeated")

# repeated value
# a = Value(-2, label="a")
# b = Value(3, label="b")
# d = a * b; d.label="d"
# e = a + b; e.label="e"
# f = d * e; f.label="f"
# f.backward()
# visualize_graph(f, filename="repeated3")

# print(Value(2) + 1)
# print(1 + Value(3))

# print(Value(3.2) * 2)
# print(2 * Value(3.2))

# # create test for this
# print(Value(2).exp())


# # simple 2-input neuron visualization
# x1 = Value(2.0, label="x1") 
# x2 = Value(0, label="x2")

# # weights for inputs
# w1 = Value(-3.0, label="w1")
# w2 = Value(1.0, label="w2")

# # bias 
# b = Value(6.8813735870195432, label="b")

# x1w1 = x1 * w1; x1w1.label = "x1*w1"
# x2w2 = x2 * w2; x2w2.label = "x2*w2"

# x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"
# n = x1w1x2w2 + b; n.label="n"
 
# e = (2*n).exp()
# e.label = "e"

# o = (e - 1) / (e + 1)
# # o = n.tanh()
# o.label="o"

# o.backward()

# print(o.prev)


# visualize_graph(o, filename="neuron2")

# n1 = Neuron(2)
# x = [1, 2]

# l1 = Layer(2, 2)
# out = l1(x)

# print(out)

# out.backward()
# visualize_graph(out)


# l1 = Layer(3, 4)
# x1 = [1, 2, 3]
# out = l1(x1)

# print(out)
# l2 = Layer(4, 4)
# out2 = l2(out)

# print(out2)

# l3 = Layer(4, 1)
# out3 = l3(out2)

# print(out3)

# m1 = MLP(3, [4, 4, 1])
# x1 = [1, 2, 3]
# print(m1.layers)
# out_mlp = m1(x1)
# out_mlp.backward()
# visualize_graph(out_mlp)

# print(out_mlp)

lr = 0.005
m1 = MLP(3, [4, 8, 1])
xs = [[2, 3, -1], [3, -1, .5], [0.5, 1, 1], [1, 1, -1]]
y = [1, 0, 0, -1]

training_steps = 1000

for _ in range(training_steps):
    
    y_pred = [m1(x) for x in xs]    

    # mean squared error
    loss = sum((yout - ygt)**2 for yout, ygt in zip(y_pred, y))
    
    # reset gradients before backward
    m1.zero_grad()
    
    
    loss.backward()

    # update params    
    for param in m1.parameters():
        param.data += -lr * param.grad
    print(loss.data)

print(y_pred)

# visualize_graph(loss, "loss")
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




# lr = 0.001
# m1 = MLP(3, [4, 4, 1])
# xs = [[2, 3, -1], [3, -1, .5], [0.5, 1, 1], [1, 1, -1]]
# y = [4, 1, 1, -2]

# training_steps = 2000

# for _ in range(training_steps):
    
#     y_pred = [m1(x) for x in xs]    

#     # mean squared error
#     loss = sum((yout - ygt)**2 for yout, ygt in zip(y_pred, y))
    
#     # reset gradients before backward
#     m1.zero_grad()
    
#     loss.backward()

#     # update params    
#     for param in m1.parameters():
#         param.data += -lr * param.grad
#     print(loss.data)

# print(y_pred)

# visualize_graph(loss, "loss")



import random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)
random.seed(1337)

# make up a dataset

from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # make y be -1 or 1
# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')


# initialize a model 
model = MLP(2, [16, 16, 1]) # 2-layer neural network
print(model)
print("number of parameters", len(model.parameters()))

# loss function
def loss(batch_size=None):
    
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    
    # forward the model to get scores
    scores = list(map(model, inputs))
    
    # svm "max-margin" loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

total_loss, acc = loss()
print(total_loss, acc)

# optimization
for k in range(100):
    
    # forward
    total_loss, acc = loss()
    
    # backward
    model.zero_grad()
    total_loss.backward()
    
    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
        
# visualize decision boundary

h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.savefig("./visualizations/sample_plot.png")
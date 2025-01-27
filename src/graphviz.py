from graphviz import Digraph
from src.value import Value

graph = Digraph(name="visualization", filename="viz", format="png", graph_attr={"rankdir": "LR"})


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b
e.label = "e"
d = e * c
d.label = "d"

print(d.prev)
print(d.op)
print(d.label)

# root = d

graph.node(name=str(id(d)), label=f"{{{d.label} | Value: {d.data:.4f}}}", shape="record")
graph.node(name=str(id(d)) + d.op, label=d.op)
graph.edge(str(id(d)) + d.op, str(id(d)))

for child in d.prev:
    graph.node(name=str(id(child)), label=f"{{{child.label} | Value: {child.data:.4f}}}", shape="record")
    graph.edge(str(id(child)), str(id(d)) + d.op)

graph.render(directory="./images")

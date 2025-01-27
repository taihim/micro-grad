from graphviz import Digraph

from src.value import Value


def trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    """Build and return two sets of all the nodes and edges in a graph."""
    nodes, edges = set(), set()

    def build(v: Value) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value) -> Digraph:
    """Draw a visualization for the given graph using graphviz."""
    dot = Digraph(format="svg", graph_attr={"rank_dir": "LR"})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        dot.node(name=uid, label=f"{{{n.label} | data {n.data:.4f}}}", shape="record")

        if n.op:
            # if this value is the result of an operation, create an op node for it
            dot.node(name=uid + n.op, label=n.op)
            dot.edge(uid + n.op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b
e.label = "e"
d = e * c
d.label = "d"

draw_dot(d)

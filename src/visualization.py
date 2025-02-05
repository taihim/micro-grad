from pathlib import Path

from graphviz import Digraph

from src.value import Value


def visualize_graph(
    root: Value, filename: str = "viz", output_format: str = "png", path: Path = Path("./visualizations")
) -> None:
    """Generate a graph visualization and store the result in the path specified."""
    graph = Digraph(name="visualization", filename=filename, format=output_format, graph_attr={"rankdir": "LR"})

    stack = [(root, "")]
    visited = []

    while stack:
        node = stack.pop(0)
        edges = []

        graph.node(
            name=str(id(node[0])),
            label=f"{{{node[0].label} | Val: {node[0].data:.4f} | Grad: {node[0].grad:.4f}}}",
            shape="record",
        )

        if node[1]:
            graph.edge(str(id(node[0])), node[1])

        if node[0].op:
            graph.node(name=str(id(node[0])) + node[0].op, label=node[0].op)
            graph.edge(str(id(node[0])) + node[0].op, str(id(node[0])))

            new_nodes = [
                (child, str(id(node[0])) + node[0].op)
                for child in node[0].prev
                if (child, str(id(node[0])) + node[0].op) not in visited
            ]
            visited.extend(new_nodes)
            edges.extend(new_nodes)
        stack.extend(edges)

    graph.render(directory=path)

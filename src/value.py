import math
from typing import Union

from src.utils import compare_float


class Value:
    """Base Value class for micrograd."""

    def __init__(self, data: float, _children: tuple["Value", ...] = (), op: str = "", label: str = "") -> None:
        self.data = data
        self.grad = 0.0
        self.label = label
        self.prev = set(_children)
        self.op = op
        # self._backward = lambda: print(f"running empty backward for {self, self.label}")
        self._backward = lambda: None

    def __repr__(self) -> str:
        """String representation of a Value object."""
        return f"Value(data={self.data} grad={self.grad})"

    def __add__(self, other_value: Union["Value", float]) -> "Value":
        """Add two Value objects."""
        other_value = other_value if isinstance(other_value, Value) else Value(other_value)
        result = Value(self.data + other_value.data, (self, other_value), "+")

        def _backward() -> None:
            # print(f"running + backward for {self}")
            # chain rule
            # we use += to accumulate the gradients for when a Value is used multiple times in the expression graph
            self.grad += 1 * result.grad
            other_value.grad += 1 * result.grad
            # print(f"values: {self, other_value}")

        result._backward = _backward

        return result

    def __radd__(self, other_value: Union["Value", float]) -> "Value":
        """Implement radd for Value objects."""
        return self + other_value

    def __neg__(self) -> "Value":
        """Negate the Value object."""
        return self * -1

    def __sub__(self, other_value: Union["Value", float]) -> "Value":
        """Subtract a Value object from another."""
        return self + (-other_value)

    def __rsub__(self, other_value: Union["Value", float]) -> "Value":
        """Implement rsub for Value objects."""
        return other_value + (-self)

    def __mul__(self, other_value: Union["Value", float]) -> "Value":
        """Multiply two Value objects."""
        other_value = other_value if isinstance(other_value, Value) else Value(other_value)
        result = Value(self.data * other_value.data, (self, other_value), "*")

        def _backward() -> None:
            # print(f"running x backward for {self}")
            # chain rule
            self.grad += other_value.data * result.grad
            other_value.grad += self.data * result.grad
            # print(f"values: {self, other_value}")

        result._backward = _backward

        return result

    def __rmul__(self, other_value: Union["Value", float]) -> "Value":  # invoked when we do 2*Value instead of Value*2
        """Implement rmul for Value objects."""
        return self * other_value

    def __truediv__(self, other_value: Union["Value", float]) -> "Value":
        """Divide a Value by another Value."""
        return self * (other_value**-1)

    def __pow__(self, power: float) -> "Value":
        """Raise a value to an numerical power."""
        x = self.data
        result = Value(x**power, (self,), f"**{power}")

        def _backward() -> None:
            # print(f"running pow backward for {self}")
            self.grad += (power * (x ** (power - 1))) * result.grad

        result._backward = _backward

        return result

    def exp(self) -> "Value":
        """Exponentiate the data attribute of the Value object."""
        x = self.data
        result = Value(math.exp(x), (self,), "exp")

        def _backward() -> None:
            # print(f"Running backward exp for {self}")
            self.grad += result.data * result.grad
            # print(f"values: {self}")

        result._backward = _backward  # noqa: SLF001

        return result

    def tanh(self) -> "Value":
        """Apply tanh to the Value object's data."""
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        result = Value(t, (self,), "tanh")

        def _backward() -> None:
            # print(f"Running backward tanh for {self}")

            # chain rule
            self.grad += (
                1 - t**2
            ) * result.grad  # derivative of tanh(x) is 1-tanh(x)^2 and we already have tanh(x) i.e. t

        result._backward = _backward  # noqa: SLF001

        return result

    def relu(self) -> "Value":
        """Apply relu to the Value object's data."""
        x = self.data
        t = max(x, 0)
        result = Value(t, (self,), "relu")

        def _backward() -> None:
            # print(f"Running backward relu for {self}")

            # chain rule
            self.grad += (result.data > 0) * result.grad

        result._backward = _backward  # noqa: SLF001

        return result

    def backward(self) -> None:
        """Perform a backward pass through the Value object and all its children."""
        self.grad = 1.0
        visited = set()
        topo = []

        def build_topo(node: "Value") -> None:
            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        for node in reversed(topo):
            node._backward()  # type: ignore[no-untyped-call]  # noqa: SLF001

    def __eq__(self, other_value: object) -> bool:
        """Check equality of two Value objects."""
        if not isinstance(other_value, Value):
            raise NotImplementedError
        return compare_float(self.data, other_value.data)

    def __hash__(self) -> int:
        """Calculate and return the hash for a Value object."""
        return hash((self.data, id(self)))

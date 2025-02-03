from math import exp
from typing import Union

from src.utils import compare_float


class Value:
    """Base Value class for micrograd."""

    def __init__(self, data: float, _children: tuple["Value", ...] = (), _op: str = "", label: str = "") -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.prev = set(_children)
        self.op = _op
        self.label = label

    def __repr__(self) -> str:
        """String representation of a Value object."""
        return f"Value(data={self.data})"

    def __add__(self, other_value: Union["Value", float]) -> "Value":
        """Add two Value objects."""
        other_value = other_value if isinstance(other_value, Value) else Value(other_value)
        result = Value(self.data + other_value.data, (self, other_value), "+")

        def _backward() -> None:
            # chain rule
            # we use += to accumulate the gradients for when a Value is used multiple times in the expression graph
            self.grad += 1 * result.grad
            other_value.grad += 1 * result.grad

        self._backward = _backward

        return result

    def __mul__(self, other_value: Union["Value", float]) -> "Value":
        """Multiply two Value objects."""
        other_value = other_value if isinstance(other_value, Value) else Value(other_value)
        result = Value(self.data * other_value.data, (self, other_value), "*")

        def _backward() -> None:
            # chain rule
            self.grad += other_value.data * result.grad
            other_value.grad += self.data * result.grad

        self._backward = _backward

        return result

    def tanh(self) -> "Value":
        """Apply tanh to the Value object's data."""
        x = self.data
        t = (exp(2 * x) - 1) / (exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # chain rule
            self.grad += (
                1 - t**2
            ) * out.grad  # derivative of tanh(x) is 1-tanh(x)^2 and we already have tanh(x) i.e. t

        self._backward = _backward

        return out

    def backward(self) -> None:
        """Perform a backward pass through the Value object and all its children."""
        self.grad = 1.0
        stack = [self]
        while stack:
            node = stack.pop()
            if node.prev:
                stack.extend(node.prev)
            node._backward()  # type: ignore[no-untyped-call]  # noqa: SLF001

    def __eq__(self, other_value: object) -> bool:
        """Check equality of two Value objects."""
        if not isinstance(other_value, Value):
            raise NotImplementedError
        return compare_float(self.data, other_value.data)

    def __hash__(self) -> int:
        """Calculate and return the hash for a Value object."""
        return hash(self.data)

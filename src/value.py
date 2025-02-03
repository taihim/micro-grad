from math import exp
from typing import Union

from src.utils import compare_float


class Value:
    """Base Value class for micrograd."""

    def __init__(self, data: float, _children: tuple["Value", ...] = (), _op: str = "", label: str = "") -> None:
        self.data = data
        self.grad = 0.0
        self.label = label
        self.prev = set(_children)
        self.op = _op
        self._backward = lambda: print(f"Running empty backward function for {self.label}")

    def __repr__(self) -> str:
        """String representation of a Value object."""
        return f"Value(data={self.data})"

    def __add__(self, other_value: Union["Value", float]) -> "Value":
        """Add two Value objects."""
        other_value = other_value if isinstance(other_value, Value) else Value(other_value)
        result = Value(self.data + other_value.data, (self, other_value), "+")

        def _backward() -> None:
            print(f"Calling addition backward for {self.label}")
            # chain rule
            # we use += to accumulate the gradients for when a Value is used multiple times in the expression graph
            self.grad += 1 * result.grad
            other_value.grad += 1 * result.grad

            print(self.label, self.grad)
            print(other_value.label, other_value.grad)

        result._backward = _backward

        return result

    def __mul__(self, other_value: Union["Value", float]) -> "Value":
        """Multiply two Value objects."""
        other_value = other_value if isinstance(other_value, Value) else Value(other_value)
        result = Value(self.data * other_value.data, (self, other_value), "*")

        def _backward() -> None:
            print(f"Calling mul backward for {self.label}")
            # chain rule
            self.grad += other_value.data * result.grad
            other_value.grad += self.data * result.grad
            print(self.label, self.grad)
            print(other_value.label, other_value.grad)

        result._backward = _backward

        return result

    def tanh(self) -> "Value":
        """Apply tanh to the Value object's data."""
        x = self.data
        t = (exp(2 * x) - 1) / (exp(2 * x) + 1)
        result = Value(t, (self,), "tanh")

        def _backward() -> None:
            # chain rule
            self.grad += (
                1 - t**2
            ) * result.grad  # derivative of tanh(x) is 1-tanh(x)^2 and we already have tanh(x) i.e. t

        result._backward = _backward  # noqa: SLF001

        return result

    def backward(self) -> None:
        """Perform a backward pass through the Value object and all its children."""
        self.grad = 1.0
        stack = [self]
        print(f"Starting backprop from {self.label}")
        while stack:
            print(f"Backprop stack: {[s.label for s in stack]}")
            node = stack.pop(0)
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

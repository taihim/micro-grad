
from src.utils import compare_float


# todo: fix unhashable type error
class Value:
    """Base Value class for micrograd."""

    def __init__(self, data: float, _children: tuple["Value", ...] = (), _op: str = "", label: str = "") -> None:
        self.data = data
        self.prev = set(_children)
        self.op = _op
        self.label = label

    def __repr__(self) -> str:
        """String representation of a Value object."""
        return f"Value(data={self.data})"

    def __add__(self, other_value: "Value") -> "Value":
        """Add two Value objects."""
        return Value(self.data + other_value.data, (self, other_value), "+")

    def __mul__(self, other_value: "Value") -> "Value":
        """Multiply two Value objects."""
        return Value(self.data * other_value.data, (self, other_value), "*")

    def __eq__(self, other_value: object) -> bool:
        """Check equality of two Value objects."""
        if not isinstance(other_value, Value):
            raise NotImplementedError
        return compare_float(self.data, other_value.data)

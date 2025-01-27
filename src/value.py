from src.utils import compare_float


class Value:
    """Base Value class for micrograd."""

    def __init__(self, data: float) -> None:
        self.data = data

    def __repr__(self) -> str:
        """String representation of a Value object."""
        return f"Value(data={self.data})"

    def __add__(self, other_value: "Value") -> "Value":
        """Add two Value objects."""
        return Value(self.data + other_value.data)

    def __mul__(self, other_value: "Value") -> "Value":
        """Multiply two Value objects."""
        return Value(self.data * other_value.data)

    def __eq__(self, other_value: object) -> bool:
        """Check equality of two Value objects."""
        if not isinstance(other_value, Value):
            raise NotImplementedError
        return compare_float(self.data, other_value.data)

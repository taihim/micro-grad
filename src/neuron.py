import random

from src.value import Value


class Neuron:
    """Base Neuron class for micrograd."""

    def __init__(self, n_in: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: list[int]) -> int:
        """__call__ implementation for Neuron objects."""
        return 0

import random

from src.value import Value


class Neuron:
    """Base Neuron class for micrograd."""

    # todo add sigmoid/relu non linearity
    def __init__(self, n_in: int, non_lin: bool = True) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]  # noqa: S311
        self.b = Value(random.uniform(-1, 1))  # noqa: S311
        self.non_lin = non_lin

    def __call__(self, x: list[int | Value]) -> Value:
        """__call__ implementation for Neuron objects."""
        if len(x) != len(self.w):
            raise ValueError(f"Expected input of length {len(self.w)}, got {len(x)}.")

        activation = sum((xi * wi for xi, wi in zip(self.w, x, strict=False)), self.b)
        return activation.tanh() if self.non_lin else activation

    def __repr__(self) -> str:
        """__repr__ implementation for Neuron objects."""
        return f"Neuron | weight: {self.w} | bias: {self.b}"

    def parameters(self) -> list[Value]:
        """Return a list of all the weights and biases for a Neuron object."""
        return [self.b, *self.w]


class Layer:
    """Base Layer class for micrograd."""

    def __init__(self, n_in: int, n_out: int) -> None:
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x: list[int | Value]) -> list[int | Value]:
        """__call__ implementation for Layer objects."""
        return [n(x) for n in self.neurons]

    def __repr__(self) -> str:
        """__repr__ implementation for Layer objects."""
        return f"<Layer | Neurons: {len(self.neurons)}>"

    def parameters(self) -> list[Value]:
        """Return a list of all the Neuron weights and biases in the Layer."""
        return [params for neuron in self.neurons for params in neuron.parameters()]


class MLP:
    """MLP class for micrograd."""

    def __init__(self, input_len: int, layer_dims: list[int]) -> None:
        mlp_dims = [input_len, *layer_dims]
        self.layers = [Layer(mlp_dims[i], mlp_dims[i + 1]) for i in range(len(layer_dims))]

    def __call__(self, x: list[int | Value]) -> int | Value | list[int | Value]:
        """__call__ implementation for Layer objects."""
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list[Value]:
        """Return a list of all parameters in the MLP."""
        return [params for layer in self.layers for params in layer.parameters()]

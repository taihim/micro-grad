from src.constants import EPSILON


def compare_float(float1: float, float2: float) -> bool:
    """Compares two floats by checking if they are within a specified epsilon value."""
    return abs(float1 - float2) < EPSILON

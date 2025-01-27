from src.value import Value

def test_value_repr() -> None:
    v1 = Value(0.1)
    
    assert str(v1) == "Value(data=0.1)"
    

def test_value_addition() -> None:
    v1 = Value(1.2)
    v2 = Value(2.2)
    
    assert v1 + v2 == Value(data=3.4)
    
def test_value_multiplication() -> None:
    v1 = Value(3.5)
    v2 = Value(2)
    print(v1*v2)
    assert v1 * v2 == Value(7)
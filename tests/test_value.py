from src.value import Value
from src.utils import compare_float

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
    assert v1 * v2 == Value(data=7)
    
def test_value_expression() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    
    assert a*b + c == Value(data=4)
    
def test_value_exp() -> None:
    a = Value(2.0)
    b = Value(-3.1)
    
    assert compare_float(a.exp().data, 7.38905609) 
    assert compare_float(b.exp().data, 0.04504920)
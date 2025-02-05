from src.value import Value
from src.utils import compare_float

def test_value_repr() -> None:
    v1 = Value(0.1)
    
    assert str(v1) == "Value(data=0.1)"
    

def test_value_addition() -> None:
    v1 = Value(1.2)
    v2 = Value(2.2)
    
    assert v1 + v2 == Value(data=3.4)


def test_value_radd() -> None:
    v1 = Value(1.2)
    assert 2.2 + v1 == Value(data=3.4)


def test_value_sub() -> None:
    v1 = Value(1.2)
    v2 = Value(2.2)
    
    assert v1 - v2 == Value(data=-1.0) 


def test_value_rsub() -> None:
    v1 = Value(1.2)
    res = 2 - v1
    
    assert res == Value(data=0.8) 

    
def test_value_multiplication() -> None:
    v1 = Value(3.5)
    v2 = Value(2)
    assert v1 * v2 == Value(data=7)


def test_value_rmul() -> None:
    v1 = Value(3.5)
    assert 2 * v1 == Value(data=7)

    
def test_value_expression() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    
    assert a * b + c == Value(data=4)
    

def test_value_exp() -> None:
    a = Value(2.0)
    b = Value(-3.1)
    
    assert compare_float(a.exp().data, 7.38905609) 
    assert compare_float(b.exp().data, 0.04504920)
    

def test_value_pow() -> None:
    a = Value(2.0) ** 2
    b = Value(-3.5) ** 3
    
    assert compare_float(a.data, 4) 
    assert compare_float(b.data, -42.875)
    


def test_value_tanh() -> None:
    a = Value(2.1)
    b = Value(-3.12)
    
    assert compare_float(a.tanh().data, 0.970451) 
    assert compare_float(b.tanh().data, -0.996107)


def test_value_div() -> None:
    a = Value(22.0)
    assert (a / 2).data == 11.0
    assert (a / -11).data == -2.0
    

def test_value_eq() -> None:
    a = Value(2.1)
    b = Value(2.1) 
    
    assert a == b

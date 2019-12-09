import pytest
import jeeautodiff as ad
import numpy as np

def test_add():
    x = ad.Node_b(2.0) + ad.Node_b(1.0)
    assert(x.val==3)
    x=ad.Node_b(2.0)+10
    y=10+ad.Node_b(2.0)
    assert(x==y)
    with pytest.raises(ValueError):
        x+"1"


def test_sub():
    x = ad.Node_b(5.0) - ad.Node_b(3)
    assert x.val == 2


def test_sub_num():
    x = ad.Node_b(5.0) - 3
    assert x.val == 2
    with pytest.raises(ValueError):
        x-"1"


def test_rsub():
    x = 5 - ad.Node_b(2.0)
    assert x.val == 3
    with pytest.raises(ValueError):
        "1"-x


def test_mul():
    x = ad.Node_b(5.0) * ad.Node_b(4)
    assert x.val == 20
    with pytest.raises(ValueError):
        x*"1"

def test_mul_num():
    x = ad.Node_b(5.0) * 4
    assert x.val == 20


def test_rmul():
    x = 5 * ad.Node_b(4)
    assert x.val == 20
    with pytest.raises(ValueError):
        "1"*x


def test_truediv():
    x=ad.Node_b(4)
    y=ad.Node_b(2)
    temp=x/y
    assert(temp.val==2)
    with pytest.raises(ValueError):
        x/"1"

def test_truediv_num():
    x = ad.Node_b(4)
    x=x/2
    assert (x.val == 2)


def test_power():
    x=ad.Node_b(2.0)
    x=x**3
    assert(x.val==8)
    with pytest.raises(ValueError):
        x**"2"

def test_rtruediv():
    x=ad.Node_b(10)
    y=1/x
    assert(y.val==0.1)
    with pytest.raises(ValueError):
        "1"/x

def test_neg():
    x = -ad.Node_b(5.0)
    assert x.val == -5

def test_equal():
    x = ad.Node_b(2.0) + ad.Node_b(1.0)
    y = ad.Node_b(3.0)
    assert x == y

def test_back_propagation():
    x = ad.Node_b(10)
    y= ad.Node_b(1)
    f=x+y
    f.backward()
    assert x.grad==1
    assert y.grad==1

def test_add_variable():
    temp=ad.Reverse_mode(2)
    x=temp.create_variable(1)
    y=temp.create_variable(2)
    with pytest.raises(Exception):
        z=temp.create_variable(3)
    temp=ad.Reverse_mode(4)
    x,y=temp.create_variable([1,2])
    z ,a = temp.create_variable([1, 2])
    with pytest.raises(Exception):
        b,c=temp.create_variable([1, 2])

def test_eval():
    temp=ad.Reverse_mode(2)
    x=temp.create_variable(1)
    with pytest.raises(ValueError):
        z = temp.calculate_gradient(3,x)
    y=temp.create_variable(2)
    f=x**2
    value,derivative=temp.calculate_gradient(f,x)
    assert value ==1
    assert derivative[0]==2 and x.grad==0
    f=x**2+y**2
    value, derivative = temp.calculate_gradient(f, [x,y])
    assert value==5
    assert derivative[0]==2 and derivative[1]==4
    assert x.grad==0 and y.grad==0
import pytest
import jeeautodiff as ad
import numpy as np

def test_add():
    x = ad.Node(2.0) + ad.Node(1.0)
    assert(x.val==3)
    assert(x.der==2)
    x=ad.Node(2.0)+10
    y=10+ad.Node(2.0)
    assert(x==y)
    with pytest.raises(ValueError):
        x+"1"


def test_sub():
    x = ad.Node(5.0) - ad.Node(3)
    assert x.val == 2
    assert x.der == 0


def test_sub_num():
    x = ad.Node(5.0) - 3
    assert x.val == 2
    assert x.der == 1
    with pytest.raises(ValueError):
        x-"1"


def test_rsub():
    x = 5 - ad.Node(2.0)
    assert x.val == 3
    assert x.der == -1
    with pytest.raises(ValueError):
        "1"-x


def test_mul():
    x = ad.Node(5.0) * ad.Node(4)
    assert x.val == 20
    assert x.der == 9
    with pytest.raises(ValueError):
        x*"1"

def test_mul_num():
    x = ad.Node(5.0) * 4
    assert x.val == 20
    assert x.der == 4


def test_rmul():
    x = 5 * ad.Node(4,2)
    assert x.val == 20
    assert x.der == 10
    with pytest.raises(ValueError):
        "1"*x


def test_truediv():
    x=ad.Node(4,2)
    y=ad.Node(2,1)
    temp=x/y
    assert(temp.val==2)
    assert(temp.der==(4-4)/4)
    with pytest.raises(ValueError):
        x/"1"

def test_truediv_num():
    x = ad.Node(4, 2)
    x=x/2
    assert (x.val == 2)
    assert (x.der == 1)


def test_power():
    x=ad.Node(2.0,2.0)
    x=x**3
    assert(x.val==8)
    assert(x.der==24)
    with pytest.raises(ValueError):
        x**"2"

def test_rtruediv():
    x=ad.Node(10,2)
    y=1/x
    assert(y.val==0.1)
    assert (y.der == -1/100*2)
    with pytest.raises(ValueError):
        "1"/x

def test_neg():
    x = -ad.Node(5.0)
    assert x.val == -5
    assert x.der == -1

def test_equal():
    x = ad.Node(2.0) + ad.Node(1.0)
    y = ad.Node(3.0, 2.0)
    assert x == y

def test_not_equal_num():
    x = ad.Node(2.0) + 2
    y = ad.Node(4.0, 2.0)
    assert x != y

def test_not_equal_node():
    x = ad.Node(2.0) + ad.Node(4.0)
    y = ad.Node(1.0, 2.0)
    assert x != y

def test_add_variable():
    temp=ad.Autodiff(2)
    x=temp.create_variable(1)
    y=temp.create_variable(2,3)
    with pytest.raises(Exception):
        z=temp.create_variable(2,3)
    temp=ad.Autodiff(4)
    x,y=temp.create_variable([1,2])
    with pytest.raises(ValueError):
        z,a=temp.create_variable((1, 2,3),[5,6])
    z ,a = temp.create_variable([1, 2],[5,6])
    with pytest.raises(Exception):
        b,c=temp.create_variable([1, 2],[5,6])

def test_eval():
    temp=ad.Autodiff(2)
    with pytest.raises(Exception):
        z = temp.eval(3)
    x=temp.create_variable(1)
    y=temp.create_variable(2)
    f=x+y
    value,derivative=temp.eval(f)
    assert np.isclose(value,3)
    assert (derivative[0]==1 and derivative[1]==1)
    f1=x+y
    f2=x-y
    with pytest.raises(Exception):
        temp.eval([f1,2])
    value, derivative = temp.eval([f1,f2])
    assert value[0]==3 and value[1]==-1
    assert derivative[0][0]==1 and derivative[0][1]==1
    assert derivative[1][0]==1 and derivative[1][-1]==-1
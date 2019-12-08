import pytest
import numpy as np
import jeeautodiff as ad


def test_sin():
    x = ad.Node(1.0)
    x = ad.sin(x)
    assert np.isclose(x.val , np.sin(1))
    assert np.isclose(x.der ,np.cos(1))
    x=ad.Node_b(1.0)
    x=ad.sin(x)
    assert np.isclose(x.val, np.sin(1))
    x = ad.sin(np.pi)
    assert np.isclose(x ,np.sin(np.pi))
    with pytest.raises(ValueError):
        ad.sin("1")


def test_cos():
    x = ad.Node(1.0)
    x = ad.cos(x)
    assert np.isclose(x.val ,np.cos(1))
    assert np.isclose(x.der ,-np.sin(1))
    x = ad.Node_b(1.0)
    x = ad.cos(x)
    assert np.isclose(x.val, np.cos(1))
    x = ad.cos(np.pi)
    assert np.isclose(x ,np.cos(np.pi))
    with pytest.raises(ValueError):
        ad.cos("1")


def test_tan():
    x = ad.Node(1.0, 2)
    x = ad.tan(x)
    assert np.isclose(x.val ,np.tan(1))
    assert np.isclose(x.der ,2 * 1 / (np.cos(1) ** 2))
    x = ad.Node_b(1.0)
    x = ad.tan(x)
    assert np.isclose(x.val, np.tan(1))
    x = ad.tan(np.pi)
    assert np.isclose(x ,np.tan(np.pi))
    with pytest.raises(ValueError):
        ad.tan("1")


def test_exp():
    x = ad.Node(0, 2)
    x = ad.exp(x)
    assert np.isclose(x.val ,1)
    assert np.isclose(x.der ,2)
    x = ad.Node_b(0)
    x = ad.exp(x)
    assert np.isclose(x.val, 1)
    x = ad.exp(0)
    assert np.isclose(x,1)
    with pytest.raises(ValueError):
        ad.exp("1")


def test_log():
    x = ad.Node(2, 2)
    x = ad.log(x)
    assert np.isclose(x.val , np.log(2))
    assert np.isclose(x.der ,1)
    x = ad.Node_b(1.0)
    x = ad.log(x)
    assert np.isclose(x.val,0)
    x = ad.log(1)
    assert np.isclose(x,0)
    with pytest.raises(ValueError):
        ad.log("1")


def test_arcsin():
    x = ad.Node(0.5, 0.5)
    x = ad.arcsin(x)
    assert np.isclose(x.val , np.arcsin(0.5))
    assert np.isclose(x.der ,(1 / np.sqrt(0.75)) * 0.5)
    x=ad.Node_b(0.5)
    x = ad.arcsin(x)
    assert np.isclose(x.val, np.arcsin(0.5))
    x = ad.arcsin(1)
    assert np.isclose(x ,np.arcsin(1))

    with pytest.raises(ValueError):
        ad.arcsin("1")
    x=ad.Node(2, 0.5)
    with pytest.raises(ValueError):
        ad.arcsin(x)
    x = ad.Node_b(2)
    with pytest.raises(ValueError):
        ad.arcsin(x)


def test_arccos():
    x = ad.Node(0.5, 0.5)
    x = ad.arccos(x)
    assert np.isclose(x.val , np.arccos(0.5))
    assert np.isclose(x.der ,(-1 / np.sqrt(0.75)) * 0.5)
    x = ad.Node_b(0.5)
    x = ad.arccos(x)
    assert np.isclose(x.val, np.arccos(0.5))
    x = ad.arccos(1)
    assert np.isclose(x ,np.arccos(1))

    with pytest.raises(ValueError):
        ad.arccos("1")
    x=ad.Node(2, 0.5)
    with pytest.raises(ValueError):
        ad.arccos(x)
    x = ad.Node_b(2)
    with pytest.raises(ValueError):
        ad.arccos(x)


def test_arctan():
    x = ad.Node(2)
    x = ad.arctan(x)
    assert np.isclose(x.val , np.arctan(2))
    assert np.isclose(x.der , 0.2)
    x = ad.Node_b(2)
    x = ad.arctan(x)
    assert np.isclose(x.val, np.arctan(2))

    x = ad.arctan(2)
    assert np.isclose(x ,np.arctan(2))

    with pytest.raises(ValueError):
        ad.arctan("1")


def test_sinh():
    x = ad.Node(2)
    x = ad.sinh(x)
    assert np.isclose(x.val , np.sinh(2))
    assert np.isclose(x.der ,np.cosh(2))
    x = ad.Node_b(2)
    x = ad.sinh(x)
    assert np.isclose(x.val, np.sinh(2))
    x = ad.sinh(2)
    assert np.isclose(x , np.sinh(2))

    with pytest.raises(ValueError):
        ad.sinh("1")

def test_cosh():
    x = ad.Node(2)
    x = ad.cosh(x)
    assert np.isclose(x.val , np.cosh(2))
    assert np.isclose(x.der , np.sinh(2))
    x = ad.Node_b(2)
    x = ad.cosh(x)
    assert np.isclose(x.val, np.cosh(2))
    x = ad.cosh(2)
    assert np.isclose(x , np.cosh(2))

    with pytest.raises(ValueError):
        ad.cosh("1")

def test_tanh():
    x = ad.Node(2)
    x = ad.tanh(x)
    assert np.isclose(x.val ,np.tanh(2))
    assert np.isclose(x.der,1 / (np.cosh(2)**2))
    x = ad.Node_b(2)
    x = ad.tanh(x)
    assert np.isclose(x.val ,np.tanh(2))
    x = ad.tanh(2)
    assert np.isclose(x , np.tanh(2))

    with pytest.raises(ValueError):
        ad.tanh("1")

def test_sqrt():
    x = ad.Node(4)
    x = ad.sqrt(x)
    assert np.isclose(x.val , 2)
    assert np.isclose(x.der , 0.25)
    x = ad.Node_b(4)
    x = ad.sqrt(x)
    assert np.isclose(x.val, 2)
    x = ad.sqrt(4)
    assert np.isclose(x ,np.sqrt(4))

    with pytest.raises(ValueError):
        ad.sqrt("1")

def test_logistic():
    x = ad.Node(2)
    x = ad.logistic(x)
    assert np.isclose(x.val, np.exp(2) / (np.exp(2) + 1))
    assert np.isclose(x.der , np.exp(-2) / ((1 + np.exp(-2)) ** 2))
    x = ad.Node_b(2)
    x = ad.logistic(x)
    assert np.isclose(x.val, np.exp(2) / (np.exp(2) + 1))
    x = ad.logistic(2)
    assert np.isclose(x , np.exp(2) / (np.exp(2) + 1))

    with pytest.raises(ValueError):
        ad.logistic("1")

def test_logb():
    x = ad.Node(2.0, 2.0)
    a=ad.logb(np.exp(1),x)
    b=ad.log(x)
    assert(np.isclose(a.val,b.val)&np.isclose(a.der,b.der))
    x = ad.Node(8.0, 2.0)
    a=ad.logb(2,x)
    assert(np.isclose(a.val,3))
    assert(np.isclose(a.der,0.360674))
    x = ad.Node_b(8.0,)
    a = ad.logb(2, x)
    assert (np.isclose(a.val, 3))
    a=ad.logb(3,27)
    assert(np.isclose(a,3))
    with pytest.raises(ValueError):
        ad.logb(x,2)
    with pytest.raises(ValueError):
        ad.logb(2,"1")
def test_power():
    x = ad.Node(2.0, 2.0)
    a=ad.power(3,x)
    assert(np.isclose(a.val,9))
    assert(np.isclose(a.der,3**2*np.log(3)*x.der))
    x = ad.Node_b(2.0)
    a = ad.power(3, x)
    assert (np.isclose(a.val, 9))
    a=ad.power(2,5)
    assert(np.isclose(a,2**5))
    with pytest.raises(ValueError):
        ad.power(x,2)
    with pytest.raises(ValueError):
        ad.power(2,"1")
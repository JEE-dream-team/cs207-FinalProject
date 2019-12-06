import pytest
import jeeautodiff as ad


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


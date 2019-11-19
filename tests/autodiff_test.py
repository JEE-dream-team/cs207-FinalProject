import pytest
import jeeautodiff as ad


def test_add():
    x = ad.Node(2.0) + ad.Node(1.0)
    assert x.val == 3
    assert x.der == 2


def test_add_num():
    x = ad.Node(2.0) + 5
    assert x.val == 7
    assert x.der == 1


def test_radd():
    x = 5 + ad.Node(2.0)
    assert x.val == 7
    assert x.der == 1


def test_equal():
    x = ad.Node(2.0) + ad.Node(1.0)
    y = ad.Node(3.0, 2.0)
    assert x == y


def test_sub():
    x = ad.Node(5.0) - ad.Node(3)
    assert x.val == 2
    assert x.der == 0


def test_sub_num():
    x = ad.Node(5.0) - 3
    assert x.val == 2
    assert x.der == 1


def test_rsub():
    x = 5 - ad.Node(2.0)
    assert x.val == 3
    assert x.der == -1


def test_mul():
    x = ad.Node(5.0) * ad.Node(4)
    assert x.val == 20
    assert x.der == 9


def test_mul_num():
    x = ad.Node(5.0) * 4
    assert x.val == 20
    assert x.der == 4


def test_rmul():
    x = 5 * ad.Node(4)
    assert x.val == 20
    assert x.der == 5


def test_truediv():
    pass


def test_truediv_num():
    pass


def test_rtruediv():
    pass


def test_pow():
    pass


def test_pow_num():
    pass


def test_neg():
    x = -ad.Node(5.0)
    assert x.val == 5
    assert x.der == -1

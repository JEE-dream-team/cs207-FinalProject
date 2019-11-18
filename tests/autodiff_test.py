import pytest
import jeeautodiff as ad

def test_add():
    x = ad.Node(2.0) + ad.Node(1.0)
    assert(x.val==3)
    assert(x.der==2)

def test_add_num():
    x = ad.Node(2.0) + 5
    assert(x.val==7)
    assert(x.der==1)

def test_equal():
    x=ad.Node(2.0) + ad.Node(1.0)
    y=ad.Node(3.0,2.0)
    assert (x==y)





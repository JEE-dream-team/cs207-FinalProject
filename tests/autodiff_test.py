import pytest
from jeeautodiff.autodiff import *

def test_add():
    x = Node(2.0) + Node(1.0)
    assert(x.val==3)
    aseert(x.der==2)

def test_equal():
    x=Node(2.0) + Node(1.0)
    y=Node(3.0,2.0)
    assert (x==y)




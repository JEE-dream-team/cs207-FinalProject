import pytest
from jeeautodiff.autodiff import autodiff

def test_add():
    x = autodiff(2.0) + autodiff(1.0)
    y = autodiff(3.0)
    assert(x == y)

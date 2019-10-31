import pytest
from src.autodiff import autodiff

def test_add():
    x = AD.AD(2.0) + AD.AD(1.0)
    y = AD.AD(3.0)
    assert(x == y)
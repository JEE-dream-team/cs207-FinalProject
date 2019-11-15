import pytest
import numpy as np
import jeeautodiff as ad
def test_sin():
    x=ad.Node(1.0)
    x=ad.sin(x)
    assert (x.val==np.sin(1))
    assert (x.der==np.cos(1))

    x=ad.sin(np.pi)
    assert(x==np.sin(np.pi))
    with pytest.raises(ValueError):
        ad.sin("1")

def test_cos():
    x=ad.Node(1.0)
    x=ad.cos(x)
    assert (x.val==np.cos(1))
    assert (x.der==-np.sin(1))

    x=ad.cos(np.pi)
    assert(x==np.cos(np.pi))
    with pytest.raises(ValueError):
        ad.cos("1")

def test_tan():
    x=ad.Node(1.0,2)
    x=ad.tan(x)
    assert (x.val==np.tan(1))
    assert (x.der==2*1/(np.cos(1)**2))

    x=ad.tan(np.pi)
    assert(x==np.tan(np.pi))
    with pytest.raises(ValueError):
        ad.tan("1")

def test_exp():
    x=ad.Node(0,2)
    x=ad.exp(x)
    assert (x.val==1)
    assert (x.der==2)

    x=ad.exp(0)
    assert(x==1)
    with pytest.raises(ValueError):
        ad.exp("1")

def test_log():
    x=ad.Node(2,2)
    x=ad.log(x)
    assert (x.val==np.log(2))
    assert (x.der==1)

    x=ad.log(1)
    assert(x==0)
    with pytest.raises(ValueError):
        ad.log("1")


import numpy as np
from jeeautodiff.autodiff import Node


def sin(N):
    if isinstance(N, Node):
        val = np.sin(N.val)
        der = np.cos(N.val) * N.der
        return Node(val, der)

    try:
        float(N)
        return np.sin(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def cos(N):
    if isinstance(N, Node):
        val = np.cos(N.val)
        der = -(np.sin(N.val) * N.der)
        return Node(val, der)

    try:
        float(N)
        return np.cos(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def tan(N):
    if isinstance(N, Node):
        val = np.tan(N.val)
        der = 1 / (np.cos(N.val) ** 2) * N.der
        return Node(val, der)

    try:
        float(N)
        return np.tan(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def exp(N):
    if isinstance(N, Node):
        val = np.exp(N.val)
        der = np.exp(N.val) * N.der
        return Node(val, der)

    try:
        float(N)
        return np.exp(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def log(N):
    if isinstance(N, Node):
        val = np.log(N.val)
        der = 1 / N.val * N.der
        return Node(val, der)

    try:
        float(N)
        return np.log(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


# print(sin(Node(2,3)).der)

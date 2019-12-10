import numpy as np
from jeeautodiff.autodiff import Node
from jeeautodiff.reverse_mode import Node_b


def sin(N):
    """
    the sin() function
    If the input is a Node instance return Node
    if the input is a Node_b instance return  Node_b
    if a scalar n return sin(n)
    else raise an error
    """
    if isinstance(N, Node):
        val = np.sin(N.val)
        der = np.cos(N.val) * N.der
        return Node(val, der)

    if isinstance(N, Node_b):
        z = Node_b(np.sin(N.val))
        z.parents = [(N, np.cos(N.val))]
        return z

    try:
        float(N)
        return np.sin(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def cos(N):
    """
      the cos() function
      If the input is a Node instance return Node
      if the input is a Node_b instance return  Node_b
      if a scalar n return cos(n)
      else raise an error
      """
    if isinstance(N, Node):
        val = np.cos(N.val)
        der = -(np.sin(N.val) * N.der)
        return Node(val, der)

    if isinstance(N, Node_b):
        z = Node_b(np.cos(N.val))
        z.parents = [(N, -np.sin(N.val))]
        return z

    try:
        float(N)
        return np.cos(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def tan(N):
    """
      the tan() function
      If the input is a Node instance return Node
      if the input is a Node_b instance return  Node_b
      if a scalar n return tan(n)
      else raise an error
      """
    if isinstance(N, Node):
        val = np.tan(N.val)
        der = 1 / (np.cos(N.val) ** 2) * N.der
        return Node(val, der)
    if isinstance(N, Node_b):
        z = Node_b(np.tan(N.val))
        z.parents = [(N, 1 / (np.cos(N.val) ** 2))]
        return z

    try:
        float(N)
        return np.tan(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def exp(N):
    """
      the exp() function
      If the input is a Node instance return Node
      if the input is a Node_b instance return  Node_b
      if a scalar n return sin(n)
      else raise an error
      """
    if isinstance(N, Node):
        val = np.exp(N.val)
        der = np.exp(N.val) * N.der
        return Node(val, der)
    if isinstance(N, Node_b):
        z = Node_b(np.exp(N.val))
        z.parents = [(N, np.exp(N.val))]
        return z

    try:
        float(N)
        return np.exp(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def log(N):
    """
      the natural log ln function
      If the input is a Node instance return Node
      if the input is a Node_b instance return  Node_b
      if a scalar n return ln(n)
      else raise an error
      """
    if isinstance(N, Node):
        val = np.log(N.val)
        der = 1 / N.val * N.der
        return Node(val, der)

    if isinstance(N, Node_b):
        z = Node_b(np.log(N.val))
        z.parents = [(N,1 / N.val)]
        return z

    try:
        float(N)
        return np.log(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def arcsin(N):
    """
      the arcsin() function
      If the input is a Node instance return Node
      if the input is a Node_b instance return  Node_b
      if a scalar n return arcsin(n)
      else raise an error
      Pay attention that arcsin only accept value between -1 and 1
      """
    if isinstance(N, Node):
        if N.val > 1 or N.val < -1:
            raise ValueError("{} should have values and derivatives -1 <= x <= 1 for arcsin use".format(N))
        val = np.arcsin(N.val)
        der = (1 / np.sqrt(1 - N.val**2)) * N.der
        return Node(val, der)
    if isinstance(N, Node_b):
        if N.val > 1 or N.val < -1:
            raise ValueError("{} should have values and derivatives -1 <= x <= 1 for arcsin use".format(N))
        z = Node_b(np.arcsin(N.val))
        z.parents = [(N, (1 / np.sqrt(1 - N.val**2)))]
        return z

    try:
        float(N)
        return np.arcsin(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))
   

def arccos(N):
    """
      the arccos() function
      If the input is a Node instance return Node
      if the input is a Node_b instance return  Node_b
      if a scalar n return arccos(n)
      else raise an error
      Pay attention that arccos only accept value between -1 and 1
    """
    if isinstance(N, Node):
        if N.val > 1 or N.val < -1 :
            raise ValueError("{} should have values and derivatives -1 <= x <= 1 for arccos use".format(N))
        val = np.arccos(N.val)
        der = (-1 / np.sqrt(1 - N.val**2)) * N.der
        return Node(val, der)

    if isinstance(N, Node_b):
        if N.val > 1 or N.val < -1:
            raise ValueError("{} should have values and derivatives -1 <= x <= 1 for arcsin use".format(N))
        z = Node_b(np.arccos(N.val))
        z.parents = [(N, (-1 / np.sqrt(1 - N.val ** 2)))]
        return z

    try:
        float(N)
        return np.arccos(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def arctan(N):
    """
         the arctan() function
         If the input is a Node instance return Node
         if the input is a Node_b instance return  Node_b
         if a scalar n return arctan(n)
         else raise an error
         """
    if isinstance(N, Node):
        val = np.arctan(N.val)
        der = 1 / (N.val**2 + 1) * N.der
        return Node(val, der)

    if isinstance(N, Node_b):
        z = Node_b(np.arctan(N.val))
        z.parents = [ (N,1 / (N.val**2 + 1)) ]
        return z
    try:
        float(N)
        return np.arctan(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def sinh(N):
    """
         the sinh() function
         If the input is a Node instance return Node
         if the input is a Node_b instance return  Node_b
         if a scalar n return sinh(n)
         else raise an error
         """
    if isinstance(N, Node):
        val = np.sinh(N.val)
        der = np.cosh(N.val) * N.der
        return Node(val, der)

    if isinstance(N, Node_b):
        z = Node_b(np.sinh(N.val))
        z.parents = [(N, np.cosh(N.val))]
        return z
    try:
        float(N)
        return np.sinh(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def cosh(N):
    """
           the cosh() function
           If the input is a Node instance return Node
           if the input is a Node_b instance return  Node_b
           if a scalar n return cosh(n)
           else raise an error
           """
    if isinstance(N, Node):
        val = np.cosh(N.val)
        der = np.sinh(N.val) * N.der
        return Node(val, der)

    if isinstance(N, Node_b):
        z = Node_b(np.cosh(N.val))
        z.parents = [(N, np.sinh(N.val))]
        return z
    try:
        float(N)
        return np.cosh(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def tanh(N):
    """
           the tanh() function
           If the input is a Node instance return Node
           if the input is a Node_b instance return  Node_b
           if a scalar n return tanh(n)
           else raise an error
           """
    if isinstance(N, Node):
        val = np.tanh(N.val)
        der = (1-np.tanh(N.val)**2) * N.der
        return Node(val, der)

    if isinstance(N, Node_b):
        z = Node_b(np.tanh(N.val))
        z.parents = [(N, (1-np.tanh(N.val)**2))]
        return z

    try:
        float(N)
        return np.tanh(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def sqrt(N):
    """
           the sqrt() function
           If the input is a Node instance return Node
           if the input is a Node_b instance return  Node_b
           if a scalar n return sqrt(n)
           else raise an error
           """
    if isinstance(N, Node):
        val = np.sqrt(N.val)
        der = 0.5 * (N.val ** -0.5) * N.der
        return Node(val, der)

    if isinstance(N, Node_b):
        z = Node_b(np.sqrt(N.val))
        z.parents = [(N, 0.5 * (N.val ** -0.5))]
        return z

    try:
        float(N)
        return np.sqrt(N)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))


def logistic(N):
    """
           the sigmoid function
           If the input is a Node instance return Node
           if the input is a Node_b instance return  Node_b
           if a scalar n return logistic(n)
           else raise an error
           """
    if isinstance(N, Node):
        val = np.exp(N.val) / (np.exp(N.val) + 1)
        der = ( np.exp(-N.val) / (1+np.exp(-N.val))**2 ) * N.der
        return Node(val, der)

    if isinstance(N, Node_b):
        z = Node_b( np.exp(N.val) / (np.exp(N.val) + 1))
        z.parents = [(N, ( np.exp(-N.val) / (1+np.exp(-N.val))**2 ))]
        return z

    try:
        float(N)
        return np.exp(N) / (np.exp(N) + 1)
    except:
        raise ValueError("{}should either be a Node instance or a number".format(N))



def logb(b, N):
    """
            input:
            b:base,a integer or float
            N:node,node_b or scalar
           the log function for any base
           If the input is a Node instance return Node
           if the input is a Node_b instance return  Node_b
           if a scalar n return logb(n)
           else raise an error
           """
    try:
        float(b)
        if isinstance(N,Node):
            new_val = np.log(N.val)/np.log(b)
            new_der = (1/(np.log(b)*N.val))*N.der
            return Node(val=new_val,der=new_der)

        if isinstance(N, Node_b):
            z = Node_b(np.log(N.val)/np.log(b))
            z.parents = [(N, (1/(np.log(b)*N.val)))]
            return z
        try:
            float(N)
            new_val = np.log(N)/np.log(b)
            return new_val
        except:
            raise ValueError("{}should either be a Node instance or a number".format(N))
    except:
        raise ValueError("the base of a logb should be an int or a float")

def power(b,N):
    """
            input:
            b:base,a integer or float
            N:node,node_b or scalar
           the exponential unction for any base
           If the input is a Node instance return Node
           if the input is a Node_b instance return  Node_b
           if a scalar n return b**n
           else raise an error
           """
    try:
        float(b)
        if isinstance(N,Node):
            val=b**N.val
            der=val*log(b)*N.der
            return Node(val, der)

        if isinstance(N, Node_b):
            val = b ** N.val
            z = Node_b(val)
            z.parents = [(N, val*log(b))]
            return z
        try:
            float(N)
            return b**N
        except:
            raise ValueError("{}should either be a Node instance or a number".format(N))
    except:
        raise ValueError("the base of a power should be an int or a float")


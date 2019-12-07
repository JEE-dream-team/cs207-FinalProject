import numpy as np
class Reverse_mode:
    def __init__(self,dimension=1):
        self.count = 0
        self.dimension=dimension

    def create_variable(self, val):
        if self.count >= self.dimension:
            raise Exception("Can not create more variable than pre-specified value")
        try:
            float(val)
            #position of the seed
            return Node_b(val)
        except:
                #check if user trying to create more variables than pre-specified dimension
                if self.count+len(val)>self.dimension:
                    raise Exception("Can not create more variable than pre-specified value")
                #record return list of Node
                node_ls=[]
                for i in range(0,len(val)):
                    node_ls.append(Node_b(val[i]))
                return tuple(node_ls)

    def calculate_gradient(self,f,Node_ls):
        """
        input:
            f: a function (actually a Node_b instance)
            Node_ls: a single variable or a list of variable in the function
        return:
            if Node_ls a single variable x, return df/dx as a numpy 1d array
            if Node_ls a list,suppose[x,y],return a gradient in the format [df/dx,df/dy] as a numpy 1d array
        """
        if isinstance(f,Node_b):
            #output gradient,depends on the number of variable pass in in Node_ls
            root_node=f.backward()
            if type(Node_ls) is list:
                ls = []
                for i in Node_ls:
                    ls.append(i.grad)
                for k in root_node:
                    k.reset()
                return [f.val,np.array(ls)]
            else:
                grad=Node_ls.grad
                for k in root_node:
                    k.reset()
                return [f.val,np.array([grad])]
        else:
            raise ValueError("f should be a function in back_eval")

class Node_b:
    def __init__(self, data):
        self._val = float(data)
        self._grad = 0.0

        self.parents = None

    def reset(self):
        self._grad=0.0

    @property
    def val(self):
        return self._val

    @property
    def grad(self):
        return self._grad

    def __repr__(self):

        return "ad.Node_b(val={},grad={})".format(self.val, self.grad)

    def __mul__(self, other):
        if (
                isinstance(other, int)
                or isinstance(other, float)
                or isinstance(other, Node_b)
        ):
            try:
                z = Node_b(self.val * other.val)
                z.parents = [(self, other.val), (other, self.val)]
                return z
            except AttributeError:
                z = Node_b(self.val * other)
                z.parents = [(self, other)]
                return z
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if (
                isinstance(other, int)
                or isinstance(other, float)
                or isinstance(other, Node_b)
        ):
            try:
                z = Node_b(self.val / other.val)
                z.parents = [(self, 1 / other.val), (other, -self.val / (other.val) ** 2)]
                return z
            except AttributeError:
                z = Node_b(self.val / other)
                z.parents = [(self, 1 / other)]
                return z
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __rtruediv__(self, other):
        return other*(self**(-1))

    def __add__(self, other):
        if (
                isinstance(other, int)
                or isinstance(other, float)
                or isinstance(other, Node_b)
        ):
            try:
                z = Node_b(self.val + other.val)
                z.parents = [(self, 1.0), (other, 1.0)]
                return z
            except AttributeError:
                z = Node_b(self.val + other)
                z.parents = [(self, 1.0)]
                return z
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if (
                isinstance(other, int)
                or isinstance(other, float)
                or isinstance(other, Node_b)
        ):
            try:
                z = Node_b(self.val - other.val)
                z.parents = [(self, 1.0), (other, -1.0)]
                return z
            except AttributeError:
                z = Node_b(self.val - other)
                z.parents = [(self, 1.0)]
                return z
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")
    def __rsub__(self, other):
        if (
            isinstance(other, int)
            or isinstance(other, float)
        ):
            z = Node_b(other-self.val)
            z.parents=[(self, -1.0)]
            return z
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __neg__(self):
        z = Node_b(-self.val )
        z.parents = [(self, -1.0)]
        return z

    def __pow__(self, other):
        try:
            float(other)
            z = Node_b(self.val ** other)
            z.parents = [(self, other * self.val ** (other - 1))]
            return z
        except:
            raise ValueError("power input should be an int or a float")

    def __eq__(self, other):
        return self.val == other.val and self.grad == other.grad

    def backward(self, signal=1.0):
        self._grad = signal
        root_node = backprop(self, signal)
        return root_node

def backprop(node, signal):
    if node.parents is None:
        return node
    ls=[]
    for p, s in node.parents:
        new_signal = s*signal
        p._grad += new_signal
        temp=backprop(p, new_signal)
        if isinstance(temp,list):
            ls.extend(temp)
        else:
            ls.append(temp)
    return ls


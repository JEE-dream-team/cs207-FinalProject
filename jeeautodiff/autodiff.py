# from utility import sin
import numpy as np

class Autodiff:
    def __init__(self, dimension=1):
        self.count = 0
        self.dimension = dimension

    def create_variable(self, val, der=None):
        if self.count >= self.dimension:
            raise Exception("Can not create more variable than pre-specified value")
        try:
            float(val)
            #position of the seed
            position=self.count
            self.count += 1
            if der is None:
                #specify initial gradient
                gradient=np.zeros([self.dimension,])
                gradient[position]=1
                return Node(val, gradient)
            else:
                gradient = np.zeros([self.dimension, ])
                gradient[position] = der
                return Node(val,gradient)
        except:
            #user specify each seed
            if der is not None:
                if type(val) is not list or type(der) is not list:
                    raise ValueError("for multiple input, both Val and Der(if not specified) should be a list")
                #check equal length of prespecified val and der
                if len(val)!=len(der):
                    raise ValueError("The value and derivative you specified are not the same length")
                if self.count+len(val)>self.dimension:
                    raise Exception("Can not create more variable than pre-specified value")
                #record return list of Node
                node_ls=[]
                for i in range(0,len(val)):
                    position = self.count
                    self.count += 1
                    gradient = np.zeros([self.dimension, ])
                    gradient[position] = der[i]
                    node_ls.append(Node(val[i], gradient))
                return tuple(node_ls)
            # if user did not specify seed, then default gradient should be 1
            else:
                if type(val) is not list:
                    raise ValueError("for multiple input, Val should be a list")
                if self.count + len(val) > self.dimension:
                    raise Exception("Can not create more variable than pre-specified value")
                # record return list of Node
                node_ls = []
                for i in range(0, len(val)):
                    position = self.count
                    self.count += 1
                    gradient = np.zeros([self.dimension, ])
                    gradient[position] = 1
                    node_ls.append(Node(val[i], gradient))
                return tuple(node_ls)




    def eval(self, N):  # evaluate
        if isinstance(N,Node):
            try:
                return (N.val, N.der)
            except:
                gradient = np.zeros([self.dimension, ])
                return (N, gradient)
        if type(N) is list:
            return_val=[]
            return_jacobian=np.empty([0,self.dimension])
            for i in N:
                if isinstance(i,Node):
                    return_val.append(i.val)
                    gradient=i.der.reshape(1,self.dimension)
                    return_jacobian=np.concatenate((return_jacobian,gradient),axis=0)
                else:
                    return_val.append(i)
                    gradient = np.zeros([1,self.dimension])
                    return_jacobian = np.concatenate((return_jacobian, gradient), axis=0)

            return (return_val,return_jacobian)
        else:
            raise Exception("Should pass a function or a vector of function into eval")



class Node:
    def __init__(self, val, der=1.0):
        self._val = val
        self._der = der

    @property
    def val(self):
        return self._val

    @property
    def der(self):
        return self._der

    def __repr__(self):

        return "ad.Node(val={},der={})".format(self.val, self.der)

    def __add__(self, other):
        if (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, Node)
        ):
            try:
                return Node(self.val + other.val, self.der + other.der)
            except AttributeError:
                return Node(self.val + other, self.der)
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __radd__(self, other):
        if (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, Node)
        ):
            return Node(self.val + other, self.der)
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __eq__(self, other):
        return self.val == other.val and self.der == other.der

    def __sub__(self, other):
        if (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, Node)
        ):
            try:
                return Node(self.val - other.val, self.der - other.der)
            except AttributeError:
                return Node(self.val - other, self.der)
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __rsub__(self, other):
        if (
            isinstance(other, int)
            or isinstance(other, float)
        ):
            return Node(other - self.val, -self.der)
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __mul__(self, other):
        if (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, Node)
        ):
            try:
                return Node(
                    self.val * other.val, self.der * other.val + self.val * other.der
                )
            except AttributeError:
                return Node(self.val * other, self.der * other)
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __rmul__(self, other):
        if (
            isinstance(other, int)
            or isinstance(other, float)
        ):
            return Node(self.val * other, self.der * other)
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __truediv__(self, other):
        if (
            isinstance(other, int)
            or isinstance(other, float)
            or isinstance(other, Node)
        ):
            try:
                numerator = self.der * other.val - self.val * other.der
                denominator = other.val * other.val
                return Node(self.val / other.val, numerator / denominator)
            except AttributeError:
                numerator = self.der * other
                denominator = other * other
                return Node(self.val / other, numerator / denominator)
        else:
            raise ValueError("inputs should either be Node instances, ints, or floats")

    def __rtruediv__(self, other):
        return other*(self**(-1))

    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Node(self.val ** other, other*(self.val**(other-1))*self.der)
        else:
            raise ValueError("power input should be an int or a float")

    def __neg__(self):
        return Node(-self.val, -self.der)

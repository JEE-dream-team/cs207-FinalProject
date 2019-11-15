#from utility import sin

class autodiff:

    def __init__(self,dimension=1):
        self.count=0
        self.dimension=dimension

    def create_variable(self,val,der=1):
        if self.count>=self.dimension:
            raise Exception("Can not create more variable than pre-specified value")
        self.count+=1
        return Node(val,der)

    def eval(self,N): #evaluate
        try:
            return (N.val,N.der)
        except:
            return (N,0)

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

    def __add__(self, other):
        try:
            return Node(self.val + other.val, self.der+other.der)
        except AttributeError:
            return Node(self.val + other, self.der)

    def __eq__(self, other):
        return (self.val == other.val and self.der == other.der)

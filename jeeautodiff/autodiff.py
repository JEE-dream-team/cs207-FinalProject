#autodiff.py


class autodiff ():

    def __init__(self, val, der=1.0):
        self.val = val
        self.der = der

    def __add__(self, other):
        try:
            return autodiff(self.val + other.val, self.der)
        except AttributeError:
            return autodiff(self.val + other, self.der)

    def __eq__(self, other):
        return (self.val == other.val and self.der == other.der)

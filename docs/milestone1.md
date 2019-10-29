# Introduction
Automatic differentiation (AD) is a programmatic approach to finding the derivative of a given function. Automatic differentiation solves for the derivative by breaking down the function into a series of elementary arithmetic operations and functions (addition, subtraction, multiplication, division, exponentiation, log10, log2, loge, sin, cos, etc.) and applying the chain rule repeatedly. In this way, complicated derivatives can be calculated quickly and with machine precision. 

Automatic differentiation offers several concrete benefits over alternative means of differentiation. Manual differentiation is prone to mathematical errors; symbolic differentiation produces expression-swelling that is costly in memory; numerical differentiation is subject to rounding errors and inefficient scaling. For its superiority along these dimensions, automatic differentiation has become core to machine learning applications, in applications like neural networks and in tools like TensorFlow and PyTorch.

"jeeautodiff" is a python package for automatically differentiating a function input to the program. The package supports forward-mode differentiation, which means the chain rule is traversed from the innermost elementary function outward.

# Background
Forward-mode automatic differentiation solves for the derivative of a given function by decomposing the function into elementary arithmetic operations and applying the chain rule repeatedly, traversing the chain rule from inside to outside. This is represented visually via a directed graph of nodes and edges, where the nodes represent some operation done on an elementary function from the decomposition. The value at the derivative of each node are found simultaneously, and each subsequent node along the directed graph can be expressed as operands that have already been calculated in some previous node. As such, the full symbolic expression of the derivative is never found; the expressions are local to each node and simplified at each step of the way.  
 
Mathematically, the forward mode is equivalent to conducting a set of operations on dual numbers, which are each expressed in a real component plus an additional dual component, ɛ. By substituting (x + ɛ x') for x in f(x) where f is any one operation, we see that operating on x returns the value of the expression as the real component and the derivative as the dual component. As functions become nested, the chain rule is applied, which requires both the value of the function and its derivative to evaluate; that the dual numbers store both simultaneously makes this calculation more efficient.

# How to Use PackageName
### Importing the package
Users will import the AD package as a python library using:

```pip install jeeautodiff```

And then they can just import them as usual in python:

```import jeeautodiff as ad```
### Instantiating AD objects
Users will instantiate AD objects by creating instances of the relevant class and passing variables into the methods of that instance. They need to specify the number of variables they want to use in the AD object. For example:

```a=ad.autodiff(3)```

Passing a “3” into the instance then requires the user to initiate 3 variables as a dictionary. For example:

```a.create_variable({‘x’: ‘3’, ‘y’: ‘0’, ‘z’: ‘5’})```

*If the user creates more variables than what is specified in this class instance, an error will be raised*

### Evaluating Functions
After initiating variables, the user can pass the function they want to evaluate to the “evaluate” method which returns a tuple of its value and gradient. Here is an example:

Suppose user wants to pass in $exp(x)+(sin(x-y))^{2}$  they will need to do:

```tuple = a.evaluate(exp(x)+(sin(x-y))**2)```

Where “tuple” is the return value, as the tuple (function_value, gradient).

*If the user passes variables that have not been initialized, the evaluate function will raise an error.*

The user can continue to evaluate different functions using the same instance of an AD object by passing different functions to the “evaluate” method defined above.

# Software Organization
### Preliminary Directory Structure:
    jeeautodiff/
	    build/
		    lib/
	    jeeautodiff/
		    tests/
        dist/
        jeeautodiff_pkg.egg-info/
        .gitignore
        .travis.yml
        LICENSE
        README.md
        setup.config
        setup.py





# Implementation
### Core Data Structures
We will use numpy.ndarray as our core data structures in our Node class to store the gradient values. The primary reason is that it is easy to pre-define the dimension of the gradient after the user specifies how many variables they will have in the function. In addition, numpy.ndarrays are more computationally efficient than some other data structures like python lists and operate smoothly with all numpy operations.
### Class Implementation
First , we will implement an autodiff class which allows the user to specify the number of variables they want to pass in, initiate values of different variables and run the class’s methods on the variables. We will also implement a class called “Node” to represent the node in the forward mode.
Similar to the dual class we implemented in lecture, each node has its own function and gradient value. In the node class, we will overwrite the dunder methods like add, multiply, power, and divide to support these operations between our Node instances. We will overwrite all elementary functions in our package so each of them can take our Node instances as inputs, and return a new dual class instance with the updated function value and gradient value.
### Class Method and Name Attributes
Our class will have two class attributes, function value as “val” and gradient as “grad”. We will overwrite all the dunder methods like add, multiply, divide and power.
### External Dependencies 
We will rely on numpy to define our own elementary functions. We may also include scipy and sklearn for test purposes.
### Elementary Functions
We will redefine all of the elementary functions in our package. They should take our Node instance as an input and return a Node instance with the updated value and gradient. We will still rely on numpy to get the true value of these operations but we will also manually calculate their derivatives and update the gradient values like what we have done in the forward mode graph.

# Introduction
Automatic differentiation (AD) is a programmatic approach to finding the derivative of a given function. Automatic differentiation solves for the derivative by breaking down the function into a series of elementary arithmetic operations and functions (addition, subtraction, multiplication, division, exponentiation, log10, log2, loge, sin, cos, etc.) and applying the chain rule repeatedly. In this way, complicated derivatives can be calculated quickly and with machine precision. 

Automatic differentiation offers several concrete benefits over alternative means of differentiation. Manual differentiation is prone to mathematical errors; symbolic differentiation produces expression-swelling that is costly in memory; numerical differentiation is subject to rounding errors and inefficient scaling. For its superiority along these dimensions, automatic differentiation has become core to machine learning applications, in applications like neural networks and in tools like TensorFlow and PyTorch.

"jeeautodiff" is a python package for automatically differentiating a function input to the program. The package supports forward-mode differentiation, which means the chain rule is traversed from the innermost elementary function outward.

# Background
Forward-mode automatic differentiation solves for the derivative of a given function by decomposing the function into elementary arithmetic operations and elementary functions and applying the chain rule repeatedly, traversing the chain rule from inside to outside. By combining the use of the chain rule with the decomposition into to elementary functions and operations, automatic differentiation can achieve results with machine precision.

### Elementary Functions
The elementary arithmetic operations include addition, subtraction, multiplication, division, etc., and the key elementary functions include exp, log, sin, cos, etc.

### Graph Structure
This is represented visually via a directed graph of nodes and edges, where the nodes represent some operation done on an elementary function from the decomposition. The value at the derivative of each node are found simultaneously, and each subsequent node along the directed graph can be expressed as operands that have already been calculated in some previous node. As such, the full symbolic expression of the derivative is never found; the expressions are local to each node and simplified at each step of the way.  

Mechanically, the expression is first broken down into an “evaluation trace,” containing a set of xi traces or intermediary computed variables. Next, the calculation is visualized via steps in an “evaluation graph,” which reflects the computation elements with edges and nodes.
 
### Dual Numbers 
Mathematically, the forward mode is equivalent to conducting a set of operations on dual numbers, which are each expressed in a real component plus an additional dual component, ɛ. By substituting (x + ɛ x') for x in f(x) where f is any one operation, we see that operating on x returns the value of the expression as the real component and the derivative as the dual component. As functions become nested, the chain rule is applied, which requires both the value of the function and its derivative to evaluate; that the dual numbers store both simultaneously makes this calculation more efficient.
 
# How to Use PackageName
### Importing the package
After we distributed our pacakage via Pypi, Users will install the AD package as a python library using:

```pip install jeeautodiff```

However, in milestone2, user will clone it via git:

```git clone https://github.com/JEE-dream-team/cs207-FinalProject.git```

They should then go in to the directory and create a new virtual environment by:

```conda create -n env_name python=3.6 anaconda```

Activate your environment:

```source activate env_name```

Install the dependencies using

```pip install -r requirements.txt```


And then they can just import them as usual in python:

```import jeeautodiff as ad```
### Instantiating AD objects
Users will instantiate AD objects by creating instances of the relevant class and passing variables into the methods of that instance. They need to specify the number of variables they want to use in the AD object,that will specify the shape of the gradient.The default number is one. However, in milestone2, we only consider one variable For example:

```a=ad.autodiff(3)```

Passing a “3” into the instance allowes the user to initiate at most 3 variables in this autodiff instance.

They can then create variable by calling the create_variable(val,der) method, if you do not specify derivatives, the default value will be 1.Create_variable function will return a varaible that you can use later in the function

```x=a.create_variable(3)```

*If the user creates more variables than what is specified in this class instance, an error will be raised*

### Evaluating Functions
After initiating variables, the user can pass the function they want to evaluate to the “eval” method which returns a tuple of its value and gradient. Here is an example:

Suppose user wants to pass in $exp(x)+(sin(x))^{2}$  they will need to do:

```tuple = a.eval(exp(x)+(sin(x))**2)```

Where “tuple” is the return value, as the tuple (function_value, gradient).

*If the user passes variables that have not been initialized, the evaluate function will raise an error.*

The user can continue to evaluate different functions using the same instance of an AD object by passing different functions to the “evaluate” method defined above.

# Software Organization
### Directory Structure
    jeeautodiff/
        build/
            lib/
            ...
        docs/
            milestone1.md
            ...
        jeeautodiff/
            __init__.py
            jeeautodiff.py
            tests/
                __init__.py
                test_integration.py
                test_module1.py
                ...
        dist/
            jeeautodiff_pkg-0.0.1-py3-none-any.whl
            Jeeautodiff_pkg-0.0.1.tar.gz
        jeeautodiff_pkg.egg-info/
            Dependency_links.txt
            PKG-INFO
            SOURCES.txt
            top_level.txt
        .gitignore
        .travis.yml
        LICENSE
        README.md
        setup.config
        setup.py

### Modules and Functionality
jeeautodiff.py will be the module in our package that contains the autodiff class, which will take user inputs and handle automatic differentiation.

### Testing Suite
We will use TravisCI to test continuous integration and CodeCov to test for code coverage. The test suites themselves will be contained in modules in the "tests" directory.

### Distribution
We will use the Python Packaging index (PyPI) to distribute the python library. 

### Packaging
We will use Python Eggs to package our software. It will allow users to easily install, uninstall, and/or upgrade the package.

# Implementation
### Core Data Structures
We will use numpy.ndarray as our core data structures in our Node class to store the gradient values. The primary reason is that it is easy to pre-define the dimension of the gradient after the user specifies how many variables they will have in the function. In addition, numpy.ndarrays are more computationally efficient than some other data structures like python lists and operate smoothly with all numpy operations.

### Class Implementation
Just talk about we redo our dunder methods in our node class and also import a utility file to handle all other elementary functions like sin,cos etc. I think that will be fine

We implemented dunder methods in a "Node" class to represent the node in forward mode. Additionally, we implemented a utility file in a unary class to handle all other elementary functions such as sin, cos, etc.

### Class Method and Name Attributes
Our class will have two class attributes, function value as “val” and gradient as “grad”. We will overwrite all the dunder methods like add, multiply, divide and power.

### External Dependencies 
We will rely on numpy to define our own elementary functions. We may also include scipy and sklearn for test purposes(for testingroot finding algorithm later).

### Elementary Functions
We will redefine all of the elementary functions in our package. They should take our Node instance as an input and return a Node instance with the updated value and gradient. We will still rely on numpy to get the true value of these operations but we will also manually calculate their derivatives and update the gradient values like what we have done in the forward mode graph.

### Aspects not implemented

We did not implement the vector input and vector function input. For vector input, our plan is to add a dimension variable to the constructor of our Node class so that the derivative will be initialized as an numpy array of the dimension we specified. suppose dimension is d which means we have d variables. The gradient will be a numpy array of shape(d,1). For vector function input to evaluated, we will just change our evaluate function in class autodiff to handle a list of function inputs and evaluate all functions in the list

# Future Features
Next, we plan next to implement reverse-mode automatic differentation. Also known as back-propagation, reverse-mode AD is foundational to neural nets. Generally, reverse-mode is straightforward to implement and understand, but is not as efficient on memory. We need to extend the AD class functionality and build out additional documentation.



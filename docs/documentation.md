# Content
- [Introduction](#Introduction)
- [Background](#Background)
    - [Overview](#Overview)
    - [Forward-mode](#Forward-mode)
    - [Reverse-mode](#Reverse-mode)
- [Usage](#Usage)
    - [Installation](#Installation)
    - [How to use](#How-to-use-jeeautodiff)
- [Software organization](#Software-Organization)
    - [Directory structure](#Directory-structure)
    - [Modules and functionality](#Modules-and-functionality)
    - [Testing suite](#Testing-suite)
    - [Distribution](#Distribution)
    - [Packaging](#Packaging)
- [Implementation Details](#Implementation-Details)
    - [Core data structures](#Core-Data-Structures)
    - [Class implementation](#Class-Implementation)
    - [External dependencies](#External-Dependencies)
    - [Elementary functions](#Elementary-Functions)
- [Future Improvements](#Future-improvements-and-possible-extensions)

# Introduction
Automatic differentiation (AD) is a powerful programmatic approach to finding the derivative of a given function. Automatic differentiation solves for the derivative by breaking down the function into a series of elementary arithmetic operations and functions (addition, subtraction, multiplication, division, exponentiation, log10, log2, loge, sin, cos, etc.) and applying the chain rule repeatedly. In this way, complicated derivatives can be calculated quickly and with machine precision.

`jeeautodiff` is a python package for automatically differentiating a function input to the program. The package supports both forward-mode differentiation and reverse-mode differentiation; descriptions and advantages of each are discussed in the [Background](#Background) section below.

# Background
## Overview
Automatic differentiation breaks down arbitrarily complicated differentiation problems into a combination of elementary arithmetic operations and elementary functions, and then applies the chain rule repeatedly to find the derivative:

<img src="https://github.com/JEE-dream-team/cs207-FinalProject/blob/final/docs/images/ChainRule.png" width="200">

Automatic differentiation offers several concrete benefits over alternative means of differentiation. Manual differentiation is prone to mathematical errors; symbolic differentiation produces expression-swelling that is costly in memory; numerical differentiation is subject to rounding errors and inefficient scaling. For its superiority along these dimensions, automatic differentiation has become core to machine learning applications, in applications like neural networks and in tools like TensorFlow and PyTorch. There are two modes of automatic differentation: forward and reverse.

## Forward mode
Forward-mode automatic differentiation solves for the derivative of a given function by decomposing the function into elementary arithmetic operations and elementary functions and applying the chain rule repeatedly, traversing the chain rule from inside to outside. By combining the use of the chain rule with the decomposition into to elementary functions and operations, automatic differentiation can achieve results with machine precision.

### Graph structure
This is represented visually via a directed graph of nodes and edges, where the nodes represent some operation done on an elementary function from the decomposition. The value at the derivative of each node are found simultaneously, and each subsequent node along the directed graph can be expressed as operands that have already been calculated in some previous node. As such, the full symbolic expression of the derivative is never found; the expressions are local to each node and simplified at each step of the way.

Mechanically, the expression is first broken down into an “evaluation trace,” containing a set of xi traces or intermediary computed variables. Next, the calculation is visualized via steps in an “evaluation graph,” which reflects the computation elements with edges and nodes.

### Dual numbers
Mathematically, the forward mode is equivalent to conducting a set of operations on dual numbers, which are each expressed in a real component plus an additional dual component, ɛ. By substituting (x + ɛ x') for x in f(x) where f is any one operation, we see that operating on x returns the value of the expression as the real component and the derivative as the dual component. As functions become nested, the chain rule is applied, which requires both the value of the function and its derivative to evaluate; that the dual numbers store both simultaneously makes this calculation more efficient.

### Example
Consider an example function:

<img src="https://github.com/JEE-dream-team/cs207-FinalProject/blob/final/docs/images/ExampleFunction.png" width="200">

Considering this complicated function becomes even more unwieldy if we were to ask how it changes with respect to x<sub>2</sub>. Expressed as a partial derivative, that becomes:

<img src="https://github.com/JEE-dream-team/cs207-FinalProject/blob/final/docs/images/ExamplePartial.png" width="300">

Forward-mode AD, however, breaks the function down into the combination of basic arithmetic functions and operations. The connection between these foundational blocks is visualized in the computational graph below:

<img src="https://github.com/JEE-dream-team/cs207-FinalProject/blob/final/docs/images/ForwardGraph.png" width="850">

To find the partial derivative with respect to x<sub>2</sub>, we find the value of the expression at each node, and also the value of the derivative of each node. In forward-mode, each term is calculated from operands that have already been evaluated in a preceding term. The left column is the **evaluation trace** and the right column is the **derivative trace**:

<img src="https://github.com/JEE-dream-team/cs207-FinalProject/blob/final/docs/images/ForwardTrace.png" width="650">

As you can see, we arrive at a single expression for the partial derivative in question. Extrapolating this approach, you can see that to calculate the gradient with respect to a total of N parameters, one would need N forward mode differentiations; thus, Jacobians are used for multivariable evaluations.

## Reverse mode
When the network becomes very big, forward mode AD can be computationally expensive. Reverse-mode AD is a specific implementation of automatic differentiation in which the computational graph is traversed in reverse. Mechanically, reverse mode calculates derivatives as the second step of a two-part process. First, the original function code is evaluated forward, populating intermediate variables and recording the dependencies in the computational graph. Next, derivatives are calculated by propagating adjoint derivatives in reverse, from the outputs to the inputs. Here, we can no longer interleave the calculation of the derivates with the evaluation of the original expression; the dual number mental model does not apply. 

Back propagation is a special case of reverse mode AD, in which the objective function is a scalar function which represents an error between the output and a true value. Back propagation is a mainstay of machine learning, as it is a common tool for training neural networks.

Let’s consider the example reviewed above.

### Example:

<img src="https://github.com/JEE-dream-team/cs207-FinalProject/blob/final/docs/images/ExampleFunction.png" width="200">

Reverse-mode, as described above, first enumerates the **evaluation trace** in the forward direction; you can see the evaluation trace in the left column is the same the evaluation trace of the forward-mode approach. The **derivative trace**, however, is evaluated from bottom-to-top. Each lower term is calcuated before the higher term. 

<img src="https://github.com/JEE-dream-team/cs207-FinalProject/blob/final/docs/images/ReverseTrace.png" width="650">

Like forward-mode, the chain rule stipulates that we again only care about the local derivative at each node. However, since the derivative trace is calculated in reverse, we cannot calcuate the node's value and derivative simultaneously.

As a result, forward-mode is advantageous when the number of functions greatly exceeds the number of inputs evaluated and reverse-mode is superior when the number of inputs is much greater than the number of functions.

### Citations
The above illustrative graphics can be attributed to:

* CS207 Course Website, David Sondak and team: https://harvard-iacs.github.io/2019-CS207/
* _The Magic of Automatic Differentiation_, Sanyam Kapoor: https://www.sanyamkapoor.com/machine-learning/autograd-magic/


# Usage
## Installation
The package is available on `PyPI`:

```pip install jeeautodiff```

From there, users can import the library as usual:

```pip install jeeautodiff as ad```

Alternatively, users can download and run the package using the source files on GitHub, by executing:
```
git clone https://github.com/JEE-dream-team/cs207-FinalProject.git
cd cs207-FinalProject
python setup.py
```

## How to use jeeautodiff
PLACEHOLDER

# Software organization
### Directory Structure
```
    jeeautodiff/
        build/
            lib/
            ...
        docs/
            documentation.md
            milestone1.md
            milestone2.md
        jeeautodiff/
            __init__.py
            autodiff.py
            reverse_mode.py
            utility.py
        tests/
            __init__.py
            autodiff_test.py
            reverse_mode_test.py
            utility_test.py
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
        requirements.txt
        setup.config
        setup.py
```

## Modules and Functionality
`jeeautodiff.py` is the module in our package that contains both forward-mode and reverse-mode functionality.

### Forward-mode functionality
The **Autodiff** class, found in `autodiff.py`, encapsulates the forward-mode automatic differentiation functionality. This file also contains the **Node** class. The user interacts directly with the main **Autodiff** class by supplying variables and expressions to be evalauted. The **Node** class overrides the standard dunder methods to represent the node in forward mode. See [Implementation Details](#Implementation-Details) for more information.

### Reverse-mode functionality
The **Reverse_mode** class, found in `reverse_mode.py`, encapsulates the reverse-mode automatic differentiation functionality. This file also contains the **Node_b** class. The user interacts directly with the main **Reverse_mode** class by supplying variables and expressions to be evalauted. The **Node_b** class overrides the standard dunder methods to represent the node in reverse mode. See [Implementation Details](#Implementation-Details) for more information.

### Utility file
`utility.py` handles all elementary functions not addressed in `autodiff.py` or `reverse_mode.py`. The functions are handled differently based on whether reverse-mode or forward-mode automatic differentiation is being used. We implemented the following functions:

**Trigonometric functions**
  - sin(x)
  - cos(x)
  - tan(x)
  - arcsin(x)
  - arcos(x)
  - arctan(x)

**Hyperbolic functions**
  - sinh(x)
  - cosh(x)
  - tanh(x)

**Logarithmic functions**
  - log(x)
  - logb(b,x)
  - logistic(x)

**Miscellaneous functions**
  - sqrt(x)
  - power(b,x)
  - exp(x)


## Testing Suite
The `jeeautodiff` package contains a test suite that runs with `pytest`. `TravisCI` tests continuous integration and `CodeCov` measures code test coverage. The test suites themselves are contained in modules in the `tests` directory, with both forward-mode and reverse-mode tests. The `TravisCI` and `CodeCov` badges are embedded in the package README.

## Distribution
The `jeeautodiff` library is available via the Python Packaging index (PyPI).

## Packaging
The software is packaged with Python Eggs, which allows users to easily install, uninstall, and/or upgrade the package.


# Implementation details
## Core Data Structures
We use  `numpy.ndarray` as our core data structure in our **Node** and **Node_b** classes to store the gradient values. The primary reason is that it is easy to pre-define the dimension of the gradient after the user specifies how many variables they will have in the function. In addition,  `numpy.ndarrays` are more computationally efficient than other data structures such as python lists, and they operate smoothly with all  `numpy` operations.

## Class Implementation
We implemented dunder methods in a **Node** class to represent the node in forward mode, and in a **Node_b** class to represent the node in reverse mode. These main classes are initiated with two attributes: val (for value) and der (for derivative), where val is required but der is optional. Standard dunder methods are re-written within the classes:
  - \_\_add__ 
  - \_\_radd__
  - \_\_mul__
  - \_\_rmul__
  - \_\_sub__
  - \_\_rsub__
  - \_\_truediv__
  - \_\_rtruediv__
  - \_\_abs__
  - \_\_neg__ (negation)
  - \_\_eq__ (comparator: equal to)
  - \_\_ne__ (comparator: not equal to)
  - \_\_pow__
   - \_\_repr__

The utility file contains all other elementary functions, as discussed [above](#Modules-and-functionality).

### Class Method and Name Attributes
Our **Node** and **Node_b** classes have two class attributes: function value as “val” and derivative as “der.” All dunder methods and their reverse equiavalents (e.g. \_\_add__ and \_\_radd__ ) are overwritten to support operations on the node's value and derivative.

## External Dependencies 
We leverage `numpy` to define some of our own elementary functions. 

## Elementary Functions
We redefined all of the elementary functions in our package. They take our **Node** instance as an input and return a **Node** instance with the updated value and gradient. We leverage `numpy` to get the true value of these operations, but we also manually calculate their derivatives and update the gradient values. The elementary arithmetic operations are redfined in the **Node** class for forward-mode AD and in the **Node_b** class for reverse-mode AD. Elementary functions are redefined in the `utility.py` file, as described [above](#Modules-and-functionality).

# Extension
The `jeeautodiff` library was extended beyond forward-mode automatic differentation to include reverse-mode automatic differentation, which was introduced in the [Background](#Background) section. 

Reverse-mode implementation details go here.

# Future improvements and possible extensions
The `jeeautodiff` library could be extended to become a more powerful tool. 

It may be useful to intelligently select reverse-mode or forward-mode automatic differentiation based on the input the user has supplied, rather than requiring them to explicitly select one mode in advance. Alternately, we could run both modes for a given input and report on the speed and memory performance of each approach. This may be helpful for developers deciding which mode to implement for a given use case.

Separately, we could build a friendlier interface for accessing the `jeeautodiff` functionality. For example, we could parse strings via the command line, and prompt users to input their functions there, rather than having them call on the library directly. We could take this one step further and build a GUI for users to execute automatic differentation. If graphical interfaces prove helpful to our userbase, we may consider graphically representing the evaluation output to elucidate the steps along the computation graph used in the automatic differentation calculations.

We may also consider extending `jeeautodiff` for particular use cases or applications; AD has previously proved useful in applications as diverse as fluid dynamics, engineering design optimization, and computational finance. And finally, we may consider extending the package to include other modes of differentation for educational purposes; for example, we could support numeric differentation. This may more directly illustrate the benefits and limitations of automatic differenation as compared to other approaches.


"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.

    Args:
        x (float): first number
        y (float): second number

    Returns:
        float: product of x and y

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
        x (float): input number

    Returns:
        float: same as input

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
        x (float): first number
        y (float): second number

    Returns:
        float: sum of x and y

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
        x (float): input number

    Returns:
        int: negative of x

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another.

    Args:
        x (float): first number
        y (float): second number

    Returns:
        bool: True if x < y, or False otherwise

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal.

    Args:
        x (float): first number
        y (float): second number

    Returns:
        bool: True if x = y, or False otherwise

    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
        x (float): first number
        y (float): second number

    Returns:
        float: larger of x and y

    """
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value.
    Close is defined as the absolute difference being less than 1e-2.

    Args:
        x (float): first number
        y (float): second number

    Returns:
        bool: True if x and y are close, or False otherwise

    """
    return abs(x - y) < 0.01


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
        x (float): input number

    Returns:
        float: sigmoid function output of x

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function.

    Args:
        x (float): input number

    Returns:
        float: ReLU function output of x

    """
    return max(0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm.

    Args:
        x (float): input number

    Returns:
        float: log of x

    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function

    Args:
        x (float): input number

    Returns:
        float: value of e^x

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal.

    Args:
        x (float): input number

    Returns:
        float: reciprocal of x

    """
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
        x (float): first number
        y (float): second number

    Returns:
        float: Derivative of log(x) times y

    """
    return 1.0 / x * y


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
        x (float): first number
        y (float): second number

    Returns:
        float: Derivative of inv(x) times y

    """
    derivative = -1.0 / x / x
    return derivative * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
        x (float): first number
        y (float): second number

    Returns:
        float: Derivative of ReLU(x) times y

    """
    if x == 0:
        raise ValueError("Derivative of ReLU(0) does not exist")
    elif x < 0:
        return 0.0
    else:
        return y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(lst: Iterable, fn: Callable) -> Iterable:
    """Applies a given function to each element of an iterable

    Args:
        lst (Iterable): an iterable of elements
        fn (Callable): a function to apply to each single element

    Returns:
        Iterable: result of fn applied to each element in lst

    """
    return [fn(i) for i in lst]


def zipWith(it1: Iterable, it2: Iterable, fn: Callable) -> Iterable:
    """Combines elements from two iterables using a given function.

    Args:
        it1 (Iterable): first iterable
        it2 (Iterable): second iterable
        fn (Callable): a function to apply to each pair of elements in it1 and it2

    Returns:
        Iterable: the result iterable

    """
    min_len = min(len(it1), len(it2))
    result = []
    for i in range(min_len):
        result.append(fn(it1[i], it2[i]))
    return result


def reduce(fn: Callable, it: Iterable, init: float) -> float:
    """Reduces an iterable to a single value using a given function

    Args:
        fn (Callable): function to reduce a value
        it (Iterable): iterable containing values
        init (float): initial value

    Returns:
        float: result value

    """
    result = init
    for ele in it:
        result = fn(result, ele)
    return result


def negList(lst: list[float]) -> list[float]:
    """Negate all elements in a list.

    Args:
        lst (list[float]): list of numbers

    Returns:
        list[float]: result list with all numbers negated

    """
    return map(lst, neg)


def addLists(lst1: list[float], lst2: list[float]) -> list[float]:
    """Add corresponding elements from two lists.

    This function assumes lst1 and lst2 have same length. If one is longer than
    the other, then the extra elements will be ignored.

    Args:
        lst1 (list[float]): first list
        lst2 (list[float]): second list

    Returns:
        list[float]: result of adding lst1 and lst2

    """
    return zipWith(lst1, lst2, add)


def sum(lst: list[float]) -> float:
    """Sum all elements in a list.

    Args:
        lst (list[float]): list of numbers

    Returns:
        float: sum of lst

    """
    return reduce(add, lst, 0.0)


def prod(lst: list[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
        lst (list[float]): list of numbers

    Returns:
        float: product of all elements in lst

    """
    return reduce(mul, lst, 1.0)

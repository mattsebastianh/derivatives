import numpy


def generate_derivatives(funcs):
    """
    Generate derivatives of functions.

    Parameters
    ----------
    funcs : list of callable
        List of functions to generate derivatives for.

    Returns
    -------
    derivatives : list of callable
        List of derivatives of input functions.
    """
    derivatives = []
    for func in funcs:
        def derivative(x, eps=1e-7, f=func):  # Capture func in default parameter
            return (f(x+eps) - f(x-eps)) / (2*eps)
        derivatives.append(derivative)
    return derivatives


def generate_second_derivatives(funcs):
    """
    Generate second derivatives of functions.

    Parameters
    ----------
    funcs : list of callable
        List of functions to generate second derivatives for.

    Returns
    -------
    second_derivatives : list of callable
        List of second derivatives of input functions.
    """
    second_derivatives = []
    for func in funcs:
        def second_derivative(x, eps=1e-7):
            return (func(x+eps) - 2*func(x) + func(x-eps)) / eps**2
        second_derivatives.append(second_derivative)
    return second_derivatives

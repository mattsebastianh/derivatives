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
        def second_derivative(x, eps=1e-5, f=func):  # Capture func in default parameter, larger eps for stability
            return (f(x+eps) - 2*f(x) + f(x-eps)) / eps**2
        second_derivatives.append(second_derivative)
    return second_derivatives

def generate_third_derivatives(funcs):
    """
    Generate third derivatives of functions.

    Parameters
    ----------
    funcs : list of callable
        List of functions to generate third derivatives for.

    Returns
    -------
    third_derivatives : list of callable
        List of third derivatives of input functions.
    """
    third_derivatives = []
    for func in funcs:
        def third_derivative(x, eps=1e-4, f=func):  # Capture func in default parameter, larger eps for stability
            # Using forward difference formula for third derivative
            return (f(x+3*eps) - 3*f(x+2*eps) + 3*f(x+eps) - f(x)) / eps**3
        third_derivatives.append(third_derivative)
    return third_derivatives
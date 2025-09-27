"""
Derivative Calculator Class

A class-based approach to numerical differentiation providing methods for
computing first, second, and third derivatives of mathematical functions.
"""

import numpy as np


class DerivativeCalculator:
    """
    A class for computing numerical derivatives of mathematical functions.
    
    This class provides methods for calculating first, second, and third derivatives
    using finite difference methods with configurable step sizes for optimal
    numerical stability.
    """
    
    def __init__(self, eps_first=1e-7, eps_second=1e-5, eps_third=1e-4):
        """
        Initialize the DerivativeCalculator with custom epsilon values.
        
        Parameters
        ----------
        eps_first : float, optional
            Step size for first derivative calculations (default: 1e-7)
        eps_second : float, optional  
            Step size for second derivative calculations (default: 1e-5)
        eps_third : float, optional
            Step size for third derivative calculations (default: 1e-4)
        """
        self.eps_first = eps_first
        self.eps_second = eps_second
        self.eps_third = eps_third
    
    def first_derivative(self, func, x, eps=None):
        """
        Calculate the first derivative of a function at point x.
        
        Uses the central difference method: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
        
        Parameters
        ----------
        func : callable
            The function to differentiate
        x : float
            Point at which to evaluate the derivative
        eps : float, optional
            Step size (default: uses instance eps_first)
            
        Returns
        -------
        float
            Numerical approximation of the first derivative
        """
        if eps is None:
            eps = self.eps_first
        return (func(x + eps) - func(x - eps)) / (2 * eps)
    
    def second_derivative(self, func, x, eps=None):
        """
        Calculate the second derivative of a function at point x.
        
        Uses the central difference method: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
        
        Parameters
        ----------
        func : callable
            The function to differentiate
        x : float
            Point at which to evaluate the derivative
        eps : float, optional
            Step size (default: uses instance eps_second)
            
        Returns
        -------
        float
            Numerical approximation of the second derivative
        """
        if eps is None:
            eps = self.eps_second
        return (func(x + eps) - 2 * func(x) + func(x - eps)) / eps**2
    
    def third_derivative(self, func, x, eps=None):
        """
        Calculate the third derivative of a function at point x.
        
        Uses forward difference method: f'''(x) ≈ [f(x+3h) - 3f(x+2h) + 3f(x+h) - f(x)] / h³
        
        Parameters
        ----------
        func : callable
            The function to differentiate
        x : float
            Point at which to evaluate the derivative
        eps : float, optional
            Step size (default: uses instance eps_third)
            
        Returns
        -------
        float
            Numerical approximation of the third derivative
        """
        if eps is None:
            eps = self.eps_third
        return (func(x + 3*eps) - 3*func(x + 2*eps) + 3*func(x + eps) - func(x)) / eps**3
    
    def all_derivatives(self, func, x, order=3):
        """
        Calculate multiple derivatives of a function at point x.
        
        Parameters
        ----------
        func : callable
            The function to differentiate
        x : float
            Point at which to evaluate derivatives
        order : int, optional
            Maximum order of derivative to calculate (1, 2, or 3, default: 3)
            
        Returns
        -------
        dict
            Dictionary containing derivatives with keys 'first', 'second', 'third'
        """
        if order < 1 or order > 3:
            raise ValueError("Order must be 1, 2, or 3")
        
        results = {}
        results['function'] = func(x)
        
        if order >= 1:
            results['first'] = self.first_derivative(func, x)
        if order >= 2:
            results['second'] = self.second_derivative(func, x)
        if order >= 3:
            results['third'] = self.third_derivative(func, x)
            
        return results
    
    def batch_derivatives(self, functions, x, derivative_order=1):
        """
        Calculate derivatives for multiple functions at the same point.
        
        Parameters
        ----------
        functions : list of callable
            List of functions to differentiate
        x : float
            Point at which to evaluate derivatives
        derivative_order : int, optional
            Order of derivative (1, 2, or 3, default: 1)
            
        Returns
        -------
        list
            List of derivative values for each function
        """
        if derivative_order == 1:
            return [self.first_derivative(func, x) for func in functions]
        elif derivative_order == 2:
            return [self.second_derivative(func, x) for func in functions]
        elif derivative_order == 3:
            return [self.third_derivative(func, x) for func in functions]
        else:
            raise ValueError("derivative_order must be 1, 2, or 3")
    
    def gradient_at_points(self, func, x_values):
        """
        Calculate the first derivative (gradient) at multiple points.
        
        Parameters
        ----------
        func : callable
            The function to differentiate
        x_values : array-like
            Points at which to evaluate the derivative
            
        Returns
        -------
        numpy.ndarray
            Array of derivative values at each point
        """
        x_values = np.array(x_values)
        return np.array([self.first_derivative(func, x) for x in x_values])
    
    def update_epsilon(self, eps_first=None, eps_second=None, eps_third=None):
        """
        Update the default epsilon values for derivative calculations.
        
        Parameters
        ----------
        eps_first : float, optional
            New step size for first derivatives
        eps_second : float, optional
            New step size for second derivatives
        eps_third : float, optional
            New step size for third derivatives
        """
        if eps_first is not None:
            self.eps_first = eps_first
        if eps_second is not None:
            self.eps_second = eps_second
        if eps_third is not None:
            self.eps_third = eps_third
    
    def get_epsilon_values(self):
        """
        Get the current epsilon values.
        
        Returns
        -------
        dict
            Dictionary with current epsilon values
        """
        return {
            'eps_first': self.eps_first,
            'eps_second': self.eps_second,
            'eps_third': self.eps_third
        }
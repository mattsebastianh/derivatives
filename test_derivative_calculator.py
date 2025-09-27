"""
Unit Tests for DerivativeCalculator Class

Comprehensive test suite for the DerivativeCalculator class covering all methods,
edge cases, numerical accuracy, and error handling.
"""

import unittest
import numpy as np
import math
from derivative_calculator import DerivativeCalculator


class TestDerivativeCalculator(unittest.TestCase):
    """Test cases for the DerivativeCalculator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calc = DerivativeCalculator()
        self.tolerance = 1e-4  # Tolerance for numerical comparisons
        
        # Define test functions with known analytical derivatives
        self.linear_func = lambda x: 2 * x + 3
        self.quadratic_func = lambda x: x**2
        self.cubic_func = lambda x: x**3
        self.quartic_func = lambda x: x**4
        self.sine_func = lambda x: np.sin(x)
        self.cosine_func = lambda x: np.cos(x)
        self.exponential_func = lambda x: np.exp(x)
        
    def test_initialization_default(self):
        """Test default initialization of DerivativeCalculator."""
        calc = DerivativeCalculator()
        self.assertEqual(calc.eps_first, 1e-7)
        self.assertEqual(calc.eps_second, 1e-5)
        self.assertEqual(calc.eps_third, 1e-4)
        
    def test_initialization_custom(self):
        """Test custom initialization with specific epsilon values."""
        calc = DerivativeCalculator(eps_first=1e-6, eps_second=1e-4, eps_third=1e-3)
        self.assertEqual(calc.eps_first, 1e-6)
        self.assertEqual(calc.eps_second, 1e-4)
        self.assertEqual(calc.eps_third, 1e-3)
    
    def test_first_derivative_linear(self):
        """Test first derivative of linear function: f(x) = 2x + 3, f'(x) = 2."""
        x_test = 5.0
        expected = 2.0
        result = self.calc.first_derivative(self.linear_func, x_test)
        self.assertAlmostEqual(result, expected, places=6)
        
    def test_first_derivative_quadratic(self):
        """Test first derivative of quadratic function: f(x) = x², f'(x) = 2x."""
        x_test = 3.0
        expected = 2 * x_test  # 6.0
        result = self.calc.first_derivative(self.quadratic_func, x_test)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)
        
    def test_first_derivative_sine(self):
        """Test first derivative of sine function: f(x) = sin(x), f'(x) = cos(x)."""
        x_test = math.pi / 4
        expected = np.cos(x_test)
        result = self.calc.first_derivative(self.sine_func, x_test)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)
        
    def test_second_derivative_quadratic(self):
        """Test second derivative of quadratic function: f(x) = x², f''(x) = 2."""
        x_test = 4.0
        expected = 2.0
        result = self.calc.second_derivative(self.quadratic_func, x_test)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)
        
    def test_second_derivative_cubic(self):
        """Test second derivative of cubic function: f(x) = x³, f''(x) = 6x."""
        x_test = 2.0
        expected = 6 * x_test  # 12.0
        result = self.calc.second_derivative(self.cubic_func, x_test)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)
        
    def test_second_derivative_sine(self):
        """Test second derivative of sine function: f(x) = sin(x), f''(x) = -sin(x)."""
        x_test = math.pi / 6
        expected = -np.sin(x_test)
        result = self.calc.second_derivative(self.sine_func, x_test)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)
        
    def test_third_derivative_cubic(self):
        """Test third derivative of cubic function: f(x) = x³, f'''(x) = 6."""
        x_test = 1.5
        expected = 6.0
        result = self.calc.third_derivative(self.cubic_func, x_test)
        self.assertAlmostEqual(result, expected, delta=0.1)  # Larger tolerance for 3rd derivative
        
    def test_third_derivative_quartic(self):
        """Test third derivative of quartic function: f(x) = x⁴, f'''(x) = 24x."""
        x_test = 2.0
        expected = 24 * x_test  # 48.0
        result = self.calc.third_derivative(self.quartic_func, x_test)
        self.assertAlmostEqual(result, expected, delta=1.0)  # Larger tolerance for 3rd derivative
        
    def test_third_derivative_sine(self):
        """Test third derivative of sine function: f(x) = sin(x), f'''(x) = -cos(x)."""
        x_test = math.pi / 3
        expected = -np.cos(x_test)
        result = self.calc.third_derivative(self.sine_func, x_test)
        self.assertAlmostEqual(result, expected, delta=0.1)
        
    def test_all_derivatives_order_1(self):
        """Test all_derivatives method with order=1."""
        x_test = 2.0
        result = self.calc.all_derivatives(self.quadratic_func, x_test, order=1)
        
        self.assertIn('function', result)
        self.assertIn('first', result)
        self.assertNotIn('second', result)
        self.assertNotIn('third', result)
        
        self.assertEqual(result['function'], 4.0)  # 2² = 4
        self.assertAlmostEqual(result['first'], 4.0, delta=self.tolerance)  # 2*2 = 4
        
    def test_all_derivatives_order_2(self):
        """Test all_derivatives method with order=2."""
        x_test = 3.0
        result = self.calc.all_derivatives(self.cubic_func, x_test, order=2)
        
        self.assertIn('function', result)
        self.assertIn('first', result)
        self.assertIn('second', result)
        self.assertNotIn('third', result)
        
        self.assertEqual(result['function'], 27.0)  # 3³ = 27
        self.assertAlmostEqual(result['first'], 27.0, delta=self.tolerance)  # 3*3² = 27
        self.assertAlmostEqual(result['second'], 18.0, delta=self.tolerance)  # 6*3 = 18
        
    def test_all_derivatives_order_3(self):
        """Test all_derivatives method with order=3."""
        x_test = 2.0
        result = self.calc.all_derivatives(self.cubic_func, x_test, order=3)
        
        self.assertIn('function', result)
        self.assertIn('first', result)
        self.assertIn('second', result)
        self.assertIn('third', result)
        
        self.assertEqual(result['function'], 8.0)  # 2³ = 8
        self.assertAlmostEqual(result['first'], 12.0, delta=self.tolerance)  # 3*2² = 12
        self.assertAlmostEqual(result['second'], 12.0, delta=self.tolerance)  # 6*2 = 12
        self.assertAlmostEqual(result['third'], 6.0, delta=0.1)  # 6
        
    def test_all_derivatives_invalid_order(self):
        """Test all_derivatives method with invalid order."""
        with self.assertRaises(ValueError):
            self.calc.all_derivatives(self.quadratic_func, 1.0, order=0)
        with self.assertRaises(ValueError):
            self.calc.all_derivatives(self.quadratic_func, 1.0, order=4)
            
    def test_batch_derivatives_first_order(self):
        """Test batch_derivatives method for first-order derivatives."""
        functions = [self.linear_func, self.quadratic_func, self.cubic_func]
        x_test = 2.0
        result = self.calc.batch_derivatives(functions, x_test, derivative_order=1)
        
        expected = [2.0, 4.0, 12.0]  # f'(2) for each function
        self.assertEqual(len(result), 3)
        for i, expected_val in enumerate(expected):
            self.assertAlmostEqual(result[i], expected_val, delta=self.tolerance)
            
    def test_batch_derivatives_second_order(self):
        """Test batch_derivatives method for second-order derivatives."""
        functions = [self.quadratic_func, self.cubic_func]
        x_test = 3.0
        result = self.calc.batch_derivatives(functions, x_test, derivative_order=2)
        
        expected = [2.0, 18.0]  # f''(3) for each function
        self.assertEqual(len(result), 2)
        for i, expected_val in enumerate(expected):
            self.assertAlmostEqual(result[i], expected_val, delta=self.tolerance)
            
    def test_batch_derivatives_invalid_order(self):
        """Test batch_derivatives method with invalid derivative order."""
        functions = [self.linear_func]
        with self.assertRaises(ValueError):
            self.calc.batch_derivatives(functions, 1.0, derivative_order=0)
        with self.assertRaises(ValueError):
            self.calc.batch_derivatives(functions, 1.0, derivative_order=4)
            
    def test_gradient_at_points(self):
        """Test gradient_at_points method."""
        x_values = [1.0, 2.0, 3.0, 4.0]
        result = self.calc.gradient_at_points(self.quadratic_func, x_values)
        
        expected = np.array([2.0, 4.0, 6.0, 8.0])  # 2x for x in x_values
        self.assertEqual(len(result), 4)
        np.testing.assert_allclose(result, expected, atol=self.tolerance)
        
    def test_gradient_at_points_numpy_array_input(self):
        """Test gradient_at_points method with numpy array input."""
        x_values = np.array([0.5, 1.5, 2.5])
        result = self.calc.gradient_at_points(self.sine_func, x_values)
        
        expected = np.cos(x_values)
        np.testing.assert_allclose(result, expected, atol=self.tolerance)
        
    def test_update_epsilon(self):
        """Test update_epsilon method."""
        # Test updating individual epsilon values
        self.calc.update_epsilon(eps_first=1e-8)
        self.assertEqual(self.calc.eps_first, 1e-8)
        self.assertEqual(self.calc.eps_second, 1e-5)  # Should remain unchanged
        
        self.calc.update_epsilon(eps_second=1e-6, eps_third=1e-3)
        self.assertEqual(self.calc.eps_first, 1e-8)  # Should remain unchanged
        self.assertEqual(self.calc.eps_second, 1e-6)
        self.assertEqual(self.calc.eps_third, 1e-3)
        
    def test_get_epsilon_values(self):
        """Test get_epsilon_values method."""
        epsilon_values = self.calc.get_epsilon_values()
        
        expected_keys = ['eps_first', 'eps_second', 'eps_third']
        for key in expected_keys:
            self.assertIn(key, epsilon_values)
            
        self.assertEqual(epsilon_values['eps_first'], 1e-7)
        self.assertEqual(epsilon_values['eps_second'], 1e-5)
        self.assertEqual(epsilon_values['eps_third'], 1e-4)
        
    def test_custom_epsilon_in_methods(self):
        """Test using custom epsilon in individual derivative methods."""
        x_test = 2.0
        custom_eps = 1e-6
        
        result = self.calc.first_derivative(self.quadratic_func, x_test, eps=custom_eps)
        expected = 4.0
        self.assertAlmostEqual(result, expected, delta=self.tolerance)
        
        result = self.calc.second_derivative(self.cubic_func, x_test, eps=custom_eps)
        expected = 12.0
        self.assertAlmostEqual(result, expected, delta=0.01)  # Larger tolerance for second derivative with custom eps
        
    def test_edge_case_zero_point(self):
        """Test derivatives at x = 0."""
        x_test = 0.0
        
        # First derivative of x² at x=0 should be 0
        result = self.calc.first_derivative(self.quadratic_func, x_test)
        self.assertAlmostEqual(result, 0.0, delta=self.tolerance)
        
        # Second derivative of x² at x=0 should be 2
        result = self.calc.second_derivative(self.quadratic_func, x_test)
        self.assertAlmostEqual(result, 2.0, delta=self.tolerance)
        
    def test_negative_x_values(self):
        """Test derivatives at negative x values."""
        x_test = -2.0
        
        # First derivative of x³ at x=-2 should be 3*(-2)² = 12
        result = self.calc.first_derivative(self.cubic_func, x_test)
        self.assertAlmostEqual(result, 12.0, delta=self.tolerance)
        
        # Second derivative of x³ at x=-2 should be 6*(-2) = -12
        result = self.calc.second_derivative(self.cubic_func, x_test)
        self.assertAlmostEqual(result, -12.0, delta=self.tolerance)


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDerivativeCalculator)
    
    # Run the tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
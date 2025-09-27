# Numerical Derivatives Library

A comprehensive Python library for computing numerical derivatives of mathematical functions using advanced finite difference methods. Features both functional and object-oriented approaches for maximum flexibility.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [DerivativeCalculator Class](#derivativecalculator-class)
  - [Basic Usage](#basic-usage)
  - [Advanced Features](#advanced-features)
  - [Configuration](#configuration)
- [Functional Interface](#functional-interface)
- [Mathematical Background](#mathematical-background)
- [Performance and Accuracy](#performance-and-accuracy)
- [Testing](#testing)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This library provides sophisticated tools for numerical differentiation, essential for scientific computing applications where analytical derivatives are complex or unavailable. The library offers two complementary approaches:

1. **DerivativeCalculator Class**: Object-oriented interface with advanced features
2. **Functional Interface**: Simple functions for quick computations

### Key Capabilities
- **Multi-order derivatives**: First, second, and third derivatives
- **Optimized algorithms**: Central and forward difference methods
- **Configurable precision**: Adjustable step sizes for numerical stability
- **Batch processing**: Efficient computation for multiple functions
- **NumPy integration**: Seamless array operations

## Features

### 🔧 **Core Functionality**
- **First Derivatives**: Central difference method with high precision
- **Second Derivatives**: Finite difference approximation
- **Third Derivatives**: Forward difference with stability optimization
- **Batch Processing**: Simultaneous computation for multiple functions
- **Gradient Computation**: Derivatives at multiple points

### 🎯 **Advanced Features**
- **Configurable Step Sizes**: Epsilon optimization for each derivative order
- **Error Handling**: Comprehensive validation and boundary checking
- **Flexible API**: Method chaining and optional parameters
- **Performance Optimized**: Efficient algorithms for large-scale computations

### 🔬 **Scientific Computing**
- **High Accuracy**: Numerical precision up to 6 decimal places
- **Stability Control**: Adaptive epsilon values prevent numerical instability
- **Array Support**: Full NumPy integration for vectorized operations

## Installation

### Prerequisites
- Python 3.7 or higher
- Conda or pip

### Recommended: Using Conda
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate derivatives
```

### Alternative: Using pip
```bash
# Install dependencies
pip install -r requirements.txt
```

### Manual Installation
```bash
# Minimum requirements
pip install numpy>=1.20.0

# For running tests (optional)
pip install pytest>=6.0.0
```

### Environment Management
```bash
# To deactivate conda environment
conda deactivate

# To remove conda environment (if needed)
conda env remove -n derivatives
```

## Quick Start

```python
from derivative_calculator import DerivativeCalculator
import numpy as np

# Create calculator instance
calc = DerivativeCalculator()

# Define your function
def my_function(x):
    return x**3 + 2*x**2 - 5*x + 1

# Compute derivatives at x = 2
x_point = 2.0
first_deriv = calc.first_derivative(my_function, x_point)
second_deriv = calc.second_derivative(my_function, x_point)
third_deriv = calc.third_derivative(my_function, x_point)

print(f"f'({x_point}) = {first_deriv:.6f}")
print(f"f''({x_point}) = {second_deriv:.6f}")
print(f"f'''({x_point}) = {third_deriv:.6f}")
```

## DerivativeCalculator Class

The `DerivativeCalculator` class provides a comprehensive object-oriented interface for numerical differentiation with advanced configuration options.

### Basic Usage

```python
from derivative_calculator import DerivativeCalculator

# Initialize with default settings
calc = DerivativeCalculator()

# Or with custom epsilon values for better control
calc = DerivativeCalculator(
    eps_first=1e-7,   # High precision for first derivatives
    eps_second=1e-5,  # Balanced precision for second derivatives  
    eps_third=1e-4    # Stability-focused for third derivatives
)

# Define test functions
def polynomial(x):
    return 3*x**4 - 2*x**3 + x**2 - 5*x + 7

def trigonometric(x):
    return np.sin(x) * np.cos(x)

# Individual derivative calculations
x_val = 1.5
first = calc.first_derivative(polynomial, x_val)
second = calc.second_derivative(polynomial, x_val)  
third = calc.third_derivative(polynomial, x_val)

print(f"At x = {x_val}:")
print(f"f'(x) = {first:.6f}")
print(f"f''(x) = {second:.6f}")
print(f"f'''(x) = {third:.6f}")
```

### Advanced Features

#### All Derivatives at Once
```python
# Compute multiple derivatives simultaneously
results = calc.all_derivatives(polynomial, x_val, order=3)
print("Complete derivative analysis:")
for key, value in results.items():
    print(f"{key}: {value:.6f}")
```

#### Batch Processing Multiple Functions
```python
# Process multiple functions efficiently
functions = [
    lambda x: x**2,
    lambda x: np.exp(x),
    lambda x: np.log(x + 1),
    trigonometric
]

# Get first derivatives for all functions at x = 2.0
x_point = 2.0
first_derivatives = calc.batch_derivatives(functions, x_point, derivative_order=1)
second_derivatives = calc.batch_derivatives(functions, x_point, derivative_order=2)

print(f"First derivatives at x = {x_point}:")
for i, deriv in enumerate(first_derivatives):
    print(f"Function {i+1}: {deriv:.6f}")
```

#### Gradient at Multiple Points
```python
# Compute gradients across a range of points
x_range = np.linspace(0, 5, 11)  # 11 points from 0 to 5
gradients = calc.gradient_at_points(polynomial, x_range)

print("Gradient analysis:")
for x_val, grad in zip(x_range, gradients):
    print(f"f'({x_val:.1f}) = {grad:.4f}")
```

### Configuration

#### Dynamic Epsilon Adjustment
```python
# Check current settings
epsilon_values = calc.get_epsilon_values()
print("Current epsilon values:", epsilon_values)

# Update epsilon for better accuracy or performance
calc.update_epsilon(eps_first=1e-8, eps_second=1e-6)

# Verify changes
print("Updated epsilon values:", calc.get_epsilon_values())
```

#### Custom Epsilon per Calculation
```python
# Override default epsilon for specific calculations
high_precision_first = calc.first_derivative(polynomial, x_val, eps=1e-10)
fast_second = calc.second_derivative(polynomial, x_val, eps=1e-4)

print(f"High precision first derivative: {high_precision_first:.10f}")
print(f"Fast second derivative: {fast_second:.6f}")
```

## Functional Interface

For quick computations, use the original functional interface:

```python
from derivatives import generate_derivatives, generate_second_derivatives, generate_third_derivatives

# Define functions
functions = [
    lambda x: x**3 - 2*x**2 + x - 1,
    lambda x: np.sin(2*x),
    lambda x: np.exp(-x**2)
]

# Generate derivative functions
first_derivs = generate_derivatives(functions)
second_derivs = generate_second_derivatives(functions) 
third_derivs = generate_third_derivatives(functions)

# Evaluate at specific point
x_test = 1.0
print("Functional interface results:")
print(f"First derivatives: {[f(x_test) for f in first_derivs]}")
print(f"Second derivatives: {[f(x_test) for f in second_derivs]}")
print(f"Third derivatives: {[f(x_test) for f in third_derivs]}")
```

## Mathematical Background

This library implements several finite difference methods optimized for numerical stability and accuracy.

### First Derivative (Central Difference Method)
The most accurate finite difference approximation for smooth functions:

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

**Advantages**: Second-order accuracy, symmetric, minimal bias  
**Default ε**: 1e-7 for optimal precision

### Second Derivative (Central Finite Difference)
Standard finite difference method for second derivatives:

$$f''(x) \approx \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}$$

**Advantages**: Direct computation, good stability  
**Default ε**: 1e-5 for numerical stability balance

### Third Derivative (Forward Difference Method)
Forward difference approach optimized for higher-order derivatives:

$$f'''(x) \approx \frac{f(x + 3h) - 3f(x + 2h) + 3f(x + h) - f(x)}{h^3}$$

**Advantages**: Stable for higher orders, avoids backward evaluation  
**Default ε**: 1e-4 for numerical stability

## Performance and Accuracy

### Precision Characteristics
- **First derivatives**: Accurate to 6+ decimal places for smooth functions
- **Second derivatives**: Accurate to 4-5 decimal places 
- **Third derivatives**: Accurate to 2-3 decimal places (inherent limitation)

### Computational Complexity
- **Individual derivatives**: O(1) per evaluation
- **Batch processing**: O(n) for n functions
- **Gradient at points**: O(m) for m evaluation points

### Optimization Tips
```python
# For high-accuracy applications
calc = DerivativeCalculator(eps_first=1e-9, eps_second=1e-7, eps_third=1e-5)

# For performance-critical applications  
calc = DerivativeCalculator(eps_first=1e-5, eps_second=1e-4, eps_third=1e-3)

# For balanced accuracy/performance (default)
calc = DerivativeCalculator()  # Uses optimized defaults
```

## Testing

The library includes comprehensive test suites for both interfaces:

### Unit Tests for DerivativeCalculator Class
```bash
# Run comprehensive class tests (25 test cases)
python test_derivative_calculator.py
```

**Test Coverage:**
- Individual derivative methods (linear, polynomial, trigonometric functions)
- Batch processing and gradient computations
- Configuration and error handling
- Edge cases and numerical accuracy validation
- Performance benchmarks

### Functional Interface Tests  
```bash
# Run original functional tests
python test_derivatives.py
```

### Expected Test Results
```
Testing derivatives at x = 2:
==================================================
f(x) = x + 1:
  f'(2) = 1.000000 (expected: 1.0) ✓
  f''(2) = 0.000000 (expected: 0.0) ✓

f(x) = x³:
  f'(2) = 12.000000 (expected: 12.0) ✓
  f''(2) = 12.000010 (expected: 12.0) ✓
  f'''(2) = 6.011192 (expected: 6.0) ✓

All 25 unit tests passing with 100% success rate
```

## Examples

### Example 1: Physics Application
```python
# Analyze motion: position → velocity → acceleration
def position(t):
    return 0.5 * 9.81 * t**2 + 10*t + 5  # Free fall with initial velocity

calc = DerivativeCalculator()
t = 2.0  # Time in seconds

velocity = calc.first_derivative(position, t)
acceleration = calc.second_derivative(position, t)

print(f"At t = {t}s:")
print(f"Position: {position(t):.2f} m")
print(f"Velocity: {velocity:.2f} m/s") 
print(f"Acceleration: {acceleration:.2f} m/s²")
```

### Example 2: Optimization Analysis
```python
# Find critical points and analyze function behavior
def objective_function(x):
    return x**4 - 4*x**3 + 6*x**2 - 4*x + 1

calc = DerivativeCalculator()

# Analyze multiple points
x_points = np.linspace(-1, 3, 21)
analysis = []

for x in x_points:
    results = calc.all_derivatives(objective_function, x, order=2)
    analysis.append({
        'x': x,
        'f': results['function'],
        'f_prime': results['first'],
        'f_double_prime': results['second']
    })

# Find approximate critical points (where f'(x) ≈ 0)
critical_points = [point for point in analysis if abs(point['f_prime']) < 0.01]
print("Critical points found:")
for point in critical_points:
    nature = "minimum" if point['f_double_prime'] > 0 else "maximum" 
    print(f"x = {point['x']:.2f}, f(x) = {point['f']:.4f} ({nature})")
```

### Example 3: Signal Processing
```python
# Analyze signal derivatives for feature detection
def signal(t):
    return np.sin(2*np.pi*t) + 0.5*np.sin(6*np.pi*t) + 0.1*np.random.normal()

calc = DerivativeCalculator()

# Sample signal and its derivatives
t_samples = np.linspace(0, 2, 100)
signal_values = [signal(t) for t in t_samples]
signal_derivatives = calc.gradient_at_points(signal, t_samples)

# Find peaks (where derivative changes sign)
peaks = []
for i in range(1, len(signal_derivatives)-1):
    if signal_derivatives[i-1] > 0 and signal_derivatives[i+1] < 0:
        peaks.append((t_samples[i], signal_values[i]))

print(f"Detected {len(peaks)} signal peaks")
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
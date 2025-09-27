# Numerical Derivatives Library

A Python library for computing numerical derivatives of mathematical functions using finite difference methods.

## Project Scope

This library provides simple and efficient tools for numerical differentiation, which is useful in various scientific computing applications where analytical derivatives are difficult or impossible to compute. The library implements:

- **First-order derivatives** using central difference approximation
- **Second-order derivatives** using finite difference methods
- Support for any callable Python function
- Batch processing of multiple functions

## Features

- **Numerical First Derivatives**: Compute first derivatives using the central difference method with customizable step size
- **Numerical Second Derivatives**: Compute second derivatives using finite difference approximation
- **Batch Processing**: Process multiple functions simultaneously
- **Flexible Input**: Works with any callable Python function
- **Customizable Precision**: Adjustable step size (epsilon) for derivative calculations
- **NumPy Integration**: Compatible with NumPy arrays and functions

## Installation

### Prerequisites

- Python 3.6 or higher
- NumPy

### Setup

1. Clone this repository or download the source files
2. Ensure NumPy is installed:
   ```bash
   pip install numpy
   ```
3. Import the library in your Python scripts:
   ```python
   from derivatives import generate_derivatives, generate_second_derivatives
   ```

## Usage

### Basic Example

```python
import numpy as np
from derivatives import generate_derivatives

# Define your functions
def quadratic(x):
    return x**2

def sine_function(x):
    return np.sin(x)

# Create a list of functions
functions = [quadratic, sine_function]

# Generate derivatives
derivatives = generate_derivatives(functions)

# Evaluate derivatives at a point
x = 2.0
print(f"Derivative of x² at x={x}: {derivatives[0](x):.6f}")  # Expected: ~4.0
print(f"Derivative of sin(x) at x={x}: {derivatives[1](x):.6f}")  # Expected: ~cos(2)
```

### Second Derivatives

```python
from derivatives import generate_second_derivatives

# Using the same functions as above
second_derivatives = generate_second_derivatives(functions)

# Evaluate second derivatives
print(f"Second derivative of x² at x={x}: {second_derivatives[0](x):.6f}")  # Expected: ~2.0
print(f"Second derivative of sin(x) at x={x}: {second_derivatives[1](x):.6f}")  # Expected: ~-sin(2)
```

### Custom Step Size

```python
# Generate derivatives with custom epsilon (step size)
derivatives = generate_derivatives(functions)

# Use custom epsilon when evaluating
custom_derivative = derivatives[0](x, eps=1e-5)
```

## API Reference

### `generate_derivatives(funcs)`

Generate first-order derivatives of a list of functions.

**Parameters:**
- `funcs` (list of callable): List of functions to differentiate

**Returns:**
- `derivatives` (list of callable): List of derivative functions

Each derivative function has the signature: `derivative(x, eps=1e-7)`
- `x`: Point at which to evaluate the derivative
- `eps`: Step size for finite difference (default: 1e-7)

### `generate_second_derivatives(funcs)`

Generate second-order derivatives of a list of functions.

**Parameters:**
- `funcs` (list of callable): List of functions to differentiate twice

**Returns:**
- `second_derivatives` (list of callable): List of second derivative functions

Each second derivative function has the signature: `second_derivative(x, eps=1e-7)`

## Mathematical Background

### First Derivative (Central Difference)

The first derivative is approximated using the central difference method:

$$f'(x) \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

### Second Derivative (Finite Difference)

The second derivative is approximated using:

$$f''(x) \approx \frac{f(x + \epsilon) - 2f(x) + f(x - \epsilon)}{\epsilon^2}$$

Where $\epsilon$ is a small step size (default: 1e-7).

## Limitations

- **Numerical Precision**: Results are approximations and subject to floating-point arithmetic limitations
- **Step Size Sensitivity**: Very small epsilon values may cause numerical instability, while large values reduce accuracy
- **Function Requirements**: Input functions must be continuous and differentiable at the points of interest
- **Performance**: Numerical differentiation is computationally more expensive than analytical derivatives

## Examples and Testing

Run the included test file to see the library in action:

```bash
python test_derivatives.py
```

This will demonstrate derivative calculations for linear, quadratic, and trigonometric functions with expected vs. computed results.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
from derivatives import generate_derivatives
import numpy

# Define test functions
def linear_func(x):
    return x + 1

def quadratic_func(x):
    return x**2

def sine_func(x):
    return numpy.sin(x)

# Test functions
test_functions = [linear_func, quadratic_func, sine_func]

# Generate derivatives
derivatives = generate_derivatives(test_functions)

# Test at x = 2
x_test = 2
print(f"Testing derivatives at x = {x_test}:")
print("-" * 40)

print(f"f(x) = x + 1:")
print(f"  f({x_test}) = {linear_func(x_test)}")
print(f"  f'({x_test}) = {derivatives[0](x_test):.6f} (expected: 1.0)")

print(f"\nf(x) = x²:")
print(f"  f({x_test}) = {quadratic_func(x_test)}")
print(f"  f'({x_test}) = {derivatives[1](x_test):.6f} (expected: {2*x_test})")

print(f"\nf(x) = sin(x):")
print(f"  f({x_test}) = {sine_func(x_test):.6f}")
print(f"  f'({x_test}) = {derivatives[2](x_test):.6f} (expected: {numpy.cos(x_test):.6f})")
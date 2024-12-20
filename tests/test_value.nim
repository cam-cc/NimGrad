import unittest
import math
import ../src/engine/value

suite "Value operations and gradients":
  test "addition and multiplication":
    echo "\n=== Test: addition and multiplication ==="
    echo "About to create values..."
    let
      x = value(2.0)
      y = value(3.0)
    
    echo "Initial values:"
    echo "x = ", x
    echo "y = ", y
    
    let z = x * y + value(1.0)  # 2 * 3 + 1 = 7
    echo "Created z: ", z
    echo "About to call backward..."
    backward(z)  # Call the procedure explicitly
    echo "Called backward"
    
    echo "\nFinal gradients:"
    echo "x.grad = ", x.grad
    echo "y.grad = ", y.grad
    echo "z.grad = ", z.grad
    
    check(z.data == 7.0)
    check(x.grad == 3.0)  # dz/dx = y = 3
    check(y.grad == 2.0)  # dz/dy = x = 2
  
  test "ReLU behavior":
    let
      x = value(-2.0)
      y = value(3.0)
    
    let z = (x * y).relu()  # relu(-6) = 0
    backward(z)
    
    check(z.data == 0.0)
    check(x.grad == 0.0)  # Gradient is 0 because ReLU output is 0
    check(y.grad == 0.0)  # Gradient is 0 because ReLU output is 0
    
    # Test positive case
    let
      a = value(2.0)
      b = value(3.0)
    
    let c = (a * b).relu()  # relu(6) = 6
    backward(c)
    
    check(c.data == 6.0)
    check(a.grad == 3.0)  # Gradient flows through ReLU
    check(b.grad == 2.0)  # Gradient flows through ReLU

  test "more complex expression":
    let
      x = value(1.5)
      y = value(2.0)
    
    let
      a = x * y    # 3.0
      b = a + x    # 4.5
      c = b.relu() # 4.5
      d = c * y    # 9.0
    
    echo "\nComputation:"
    echo "a = x * y = ", a
    echo "b = a + x = ", b
    echo "c = relu(b) = ", c
    echo "d = c * y = ", d
    
    backward(d)
    
    echo "\nGradients:"
    echo "x.grad = ", x.grad, " (should be 6.0)"
    echo "y.grad = ", y.grad, " (should be 7.5)"
    
    check(d.data == 9.0)
    check abs(x.grad - 6.0) < 1e-6  # Corrected gradient for x
    check abs(y.grad - 7.5) < 1e-6  # Corrected gradient for y
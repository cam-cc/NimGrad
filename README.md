# Nimgrad

A lightweight autograd and neural network framework in Nim, inspired by micrograd. This project focuses on educational value and code clarity, making it easier to understand the fundamentals of automatic differentiation and neural networks.

## Features

- Simple scalar autograd implementation
- Clear and understandable computational graph
- Functional and composable neural network API
- Basic neural network layers and activations
- Educational examples and thorough documentation

## Installation

```bash
nimble install nimgrad
```

## Quick Start

```nim
import nimgrad

# Create some values with gradients
let 
  x = value(2.0)
  y = value(3.0)
  
# Perform operations
let z = x * y + value(1.0)

# Compute gradients
z.backward()

# Access gradients
echo x.grad  # dy/dx
echo y.grad  # dy/dy
```

## Neural Networks

```nim
import nimgrad

# Define a simple neural network
let model = sequential(@[
  linear(2, 4),
  relu(),
  linear(4, 1)
])

# Training loop
for epoch in 0..100:
  let output = model.forward(input)
  let loss = mse(output, target)
  loss.backward()
  optimizer.step()
```

## Project Structure

The project is organized into several key components:

- `engine/`: Core autograd implementation
- `nn/`: Neural network modules and layers
- `optim/`: Optimization algorithms
- `examples/`: Usage examples and tutorials

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
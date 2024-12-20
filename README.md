# NimGrad 🧠

A lightweight autograd engine and neural network framework in Nim, inspired by micrograd. This project focuses on educational value and code clarity, making it easier to understand the fundamentals of automatic differentiation and neural networks.

## ✨ Features

- 📊 Simple scalar autograd implementation
- 🔍 Clear and understandable computational graph
- 🧩 Functional and composable neural network API
- 🛠️ Basic neural network layers and activations (Linear, Conv2D, ReLU)
- 📚 Educational examples and thorough documentation

## 🚀 Installation

### Dependencies
Make sure you have Nim (>= 1.6.0) installed on your system. Then:

```bash
# Install NimGrad
nimble install nimgrad
```

## 📖 Quick Start

### Basic Autograd Example
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

### Neural Network Example
```nim
import nimgrad
import std/strformat

# Define a simple MLP for MNIST
let model = sequential(@[
  Module(initLinear(784, 128)),  # Input layer
  Module(initReLU()),           # Activation
  Module(initLinear(128, 10))   # Output layer
])

# Training loop with batching
proc train(model: Sequential, x, y: seq[seq[Value]], epochs: int = 10) =
  let learningRate = 0.01
  
  for epoch in 0..<epochs:
    let output = model.forward(x)
    let loss = crossEntropy(output, y)
    
    # Backward pass
    model.zeroGrad()
    loss.backward()
    
    # Update parameters (SGD)
    for p in model.parameters:
      p.data -= learningRate * p.grad
    
    if epoch mod 1 == 0:
      echo fmt"Epoch {epoch}: loss = {loss.data:.4f}"
```

### CNN Example
```nim
# Create a CNN for image classification
let cnn = sequential(@[
  Module(initConv2d(
    inChannels = 1,
    outChannels = 16,
    kernelSize = (height: 3, width: 3)
  )),
  Module(initReLU()),
  Module(initConv2d(
    inChannels = 16,
    outChannels = 32,
    kernelSize = (height: 3, width: 3)
  )),
  Module(initReLU()),
  Module(initLinear(32 * 26 * 26, 10))
])
```

## 📂 Project Structure

```
nimgrad/
├── src/
│   ├── engine/      # Core autograd implementation
│   │   └── value.nim
│   ├── nn/          # Neural network modules
│   │   ├── module.nim
│   │   ├── conv2d.nim
│   │   ├── linear.nim
│   │   ├── activations.nim
│   │   └── loss.nim
│   └── tensor/      # Tensor operations
│       └── tensor_ops.nim
├── examples/        # Usage examples
└── tests/          # Unit tests
```

## 📚 Documentation

### Core Components

#### Value
The fundamental unit for automatic differentiation:
```nim
type Value* = ref object
  data*: float
  grad*: float
  prev*: seq[Value]
  op*: string
  backward_fn*: proc()
```

#### Layers
- **Linear**: Fully connected layer
- **Conv2D**: 2D Convolutional layer
- **ReLU/Tanh**: Activation functions
- **Sequential**: Container for stacking layers

### Available Operations
- Basic arithmetic: `+`, `-`, `*`, `/`
- Activation functions: `relu`, `tanh`
- Loss functions: `mse`, `crossEntropy`

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

Please make sure to update tests as appropriate.

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- Built with the amazing [Nim Programming Language](https://nim-lang.org/)
- Thanks to all contributors and the Nim community

## 📞 Contact

If you have any questions or suggestions, feel free to:
- Open an issue
- Submit a pull request
- Join our discussions

## 🚧 Roadmap

- [ ] Add more optimization algorithms
- [ ] Implement batch normalization
- [ ] Add model serialization
- [ ] Improve documentation
- [ ] Add more examples
- [ ] Performance optimizations

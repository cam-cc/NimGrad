import std/[strformat]
import ../src/engine/value
import ../src/nn/[module, linear, activations, sequential]
import std/sequtils

# Create a simple neural network
proc createModel(): Sequential =
  let layers = @[
    Module(initLinear(2, 3)),    # Input layer: 2 -> 3
    Module(initReLU()),          # ReLU activation
    Module(initLinear(3, 1))     # Output layer: 3 -> 1
  ]
  result = initSequential(layers)

# Training loop
proc train(model: Sequential, x, y: seq[float], epochs: int = 100) =
  for epoch in 0 ..< epochs:
    var totalLoss = value(0.0)
    
    # Forward pass
    let inputs = x.mapIt(value(it))
    let output = model.forward(inputs)
    let target = value(y[0])
    
    # Compute MSE loss
    let loss = (output[0] - target) * (output[0] - target)
    totalLoss = totalLoss + loss
    
    # Backward pass
    model.zeroGrad()  # Zero all gradients
    loss.backward()
    
    # Update parameters (SGD)
    let learningRate = 0.05
    for p in model.parameters:
      p.data -= learningRate * p.grad
    
    if epoch mod 10 == 0:
      echo fmt"Epoch {epoch}: loss = {totalLoss.data:.4f}"

when isMainModule:
  # Create model
  let model = createModel()
  
  # Training data (XOR function)
  let
    x = @[0.0, 1.0]  # Input
    y = @[1.0]       # Target output
  
  # Train the model
  train(model, x, y)
  
  # Test the model
  let testInput = x.mapIt(value(it))
  let prediction = model.forward(testInput)
  echo fmt"Prediction: {prediction[0].data:.4f} (target: {y[0]:.4f})"
import std/[math, sequtils]
import ../engine/value
import ./module

type
  Softmax* = ref object of Module

  CrossEntropyLoss* = ref object of Module

proc initSoftmax*(): Softmax =
  result = Softmax()

proc initCrossEntropyLoss*(): CrossEntropyLoss =
  result = CrossEntropyLoss()

method forward*(self: Softmax, x: seq[Value]): seq[Value] =
  # Compute softmax: exp(xi) / sum(exp(xj))
  var maxVal = x[0].data
  for val in x[1..^1]:
    maxVal = max(maxVal, val.data)
  
  # Subtract max for numerical stability
  let expValues = x.mapIt((it.data - maxVal).exp())
  let sumExp = expValues.sum()
  
  # Convert back to Value objects with proper gradients
  result = newSeq[Value](x.len)
  for i in 0..<x.len:
    let expX = (x[i] - value(maxVal)).exp()
    result[i] = expX / value(sumExp)

method forward*(self: CrossEntropyLoss, pred, target: seq[Value]): Value =
  ## Cross Entropy Loss with built-in softmax
  # Apply softmax first
  let softmax = initSoftmax()
  let probs = softmax.forward(pred)
  
  # Compute cross entropy loss: -sum(target_i * log(prob_i))
  var loss = value(0.0)
  for i in 0..<probs.len:
    if target[i].data == 1.0:  # One-hot encoded target
      loss = loss - probs[i].log()
  
  return loss

# Convenience proc for computing loss with integer target
proc forward*(self: CrossEntropyLoss, pred: seq[Value], target: int): Value =
  var oneHot = newSeq[Value](pred.len)
  for i in 0..<pred.len:
    oneHot[i] = value(if i == target: 1.0 else: 0.0)
  
  return self.forward(pred, oneHot)

when isMainModule:
  # Test Softmax
  echo "\nTesting Softmax:"
  let 
    softmax = initSoftmax()
    input = @[value(1.0), value(2.0), value(3.0)]
    output = softmax.forward(input)
  
  echo "Input: ", input.mapIt(it.data)
  echo "Softmax output: ", output.mapIt(it.data)
  echo "Sum of probabilities: ", output.mapIt(it.data).sum()
  
  # Test CrossEntropyLoss
  echo "\nTesting CrossEntropyLoss:"
  let 
    criterion = initCrossEntropyLoss()
    logits = @[value(-1.0), value(0.0), value(2.0)]
    target = 2  # Target class is 2
  
  let loss = criterion.forward(logits, target)
  echo "Logits: ", logits.mapIt(it.data)
  echo "Target class: ", target
  echo "Loss: ", loss.data
  
  # Test backprop
  loss.backward()
  echo "Gradients: ", logits.mapIt(it.grad)
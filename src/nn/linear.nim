import std/[sequtils, random, math]
import ../engine/value
import ./module

type
  Linear* = ref object of Module
    inFeatures*: int
    outFeatures*: int
    weights*: seq[seq[Value]]
    bias*: seq[Value]

proc initLinear*(inFeatures, outFeatures: int): Linear =
  ## Initialize a new linear layer
  result = Linear(
    inFeatures: inFeatures,
    outFeatures: outFeatures
  )
  
  # Initialize weights with small random values
  randomize()
  result.weights = newSeqWith(outFeatures,
    newSeqWith(inFeatures,
      value(rand(-1.0..1.0) / sqrt(inFeatures.float))
    )
  )
  
  # Initialize biases to zero
  result.bias = newSeqWith(outFeatures, value(0.0))
  
  # Add parameters to the base class
  result.parameters = concat(
    result.weights.concat(),  
    result.bias             
  )

method forward*(self: Linear, x: seq[Value]): seq[Value] =
  ## Forward pass for linear layer
  # Check input dimension
  assert x.len == self.inFeatures, "Input size mismatch"
  
  result = newSeq[Value](self.outFeatures)
  
  # Compute output for each neuron
  for i in 0 ..< self.outFeatures:
    var sum = self.bias[i]  # Start with bias
    
    # Add weighted inputs
    for j in 0 ..< self.inFeatures:
      sum = sum + (self.weights[i][j] * x[j])
    
    result[i] = sum
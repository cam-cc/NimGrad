import ../engine/value
import ./module

type
  Sequential* = ref object of Module
    layers*: seq[Module]

proc initSequential*(layers: seq[Module]): Sequential =
  result = Sequential(layers: layers)
  
  # Collect parameters from all layers
  for layer in layers:
    result.parameters.add(layer.parameters)

method forward*(self: Sequential, x: auto): auto =
  var output = x
  for layer in self.layers:
    output = layer.forward(output)
  return output
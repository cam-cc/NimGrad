import ../engine/value

type
  Module* = ref object of RootObj
    ## Base class for all neural network modules
    parameters*: seq[Value]

method forward*(self: Module, x: seq[Value]): seq[Value] {.base.} =
  ## Base forward method - should be overridden by subclasses
  raise newException(CatchableError, "forward method not implemented")

method forward*(self: Module, x: seq[seq[seq[seq[Value]]]]): seq[seq[seq[seq[Value]]]] {.base.} =
  ## Base forward method for 4D input (batch, channels, height, width)
  raise newException(CatchableError, "forward method not implemented")

proc zeroGrad*(self: Module) =
  ## Zeros out all parameter gradients
  for p in self.parameters:
    p.grad = 0.0
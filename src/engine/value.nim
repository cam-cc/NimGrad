import std/sets
import std/hashes
import std/algorithm
import std/sequtils
import std/math

type
  Value* = ref object
    data*: float
    grad*: float
    prev*: seq[Value]
    op*: string
    backward_fn*: proc() {.closure.}

proc hash*(x: Value): Hash =
  hash(cast[pointer](x))

proc `$`*(self: Value): string =
  result = "Value(data=" & $self.data & ", grad=" & $self.grad & ", op='" & self.op & "')"

proc value*(data: float): Value =
  ## TLDR: Creates a new Value with the given data
  ## This is the constructor for the Value type. It initializes a new Value object with the given data 
  ## sets the gradient to 0, and initializes the previous values to an empty sequence.
  result = Value(
    data: data,
    grad: 0.0,
    prev: @[],
    op: ""
  )

proc `+`*(self, other: Value): Value =
  ## TLDR: Addition operation with gradient computation
  ## This is the addition operation for the Value type. It creates a new Value object with the sum of the data of the two values.
  ## It sets the gradient to 0, and initializes the previous values to an empty sequence.
  let output = Value(
    data: self.data + other.data,
    grad: 0.0,
    prev: @[self, other],
    op: "+"
  )
  output.backward_fn = proc() =
    self.grad += output.grad
    other.grad += output.grad
  return output

proc `-`*(self, other: Value): Value =
  ## TLDR: Subtraction operation with gradient computation
  ## This is the subtraction operation for the Value type. It creates a new Value object with the difference of the data of the two values.
  ## It sets the gradient to 0, and initializes the previous values to an empty sequence.
  let output = Value(
    data: self.data - other.data,
    grad: 0.0,
    prev: @[self, other],
    op: "-"
  )
  output.backward_fn = proc() =
    self.grad += output.grad
    other.grad -= output.grad
  return output

proc `*`*(self, other: Value): Value =
  ## TLDR: Multiplication operation with gradient computation
  ## This is the multiplication operation for the Value type. It creates a new Value object with the product of the data of the two values.
  ## It sets the gradient to 0, and initializes the previous values to an empty sequence.
  let output = Value(
    data: self.data * other.data,
    grad: 0.0,
    prev: @[self, other],
    op: "*"
  )
  output.backward_fn = proc() =
    self.grad += other.data * output.grad
    other.grad += self.data * output.grad
  return output

proc `/`*(self, other: Value): Value =
  ## TLDR: Division operation with gradient computation
  ## This is the division operation for the Value type. It creates a new Value object with the quotient of the data of the two values.
  ## It sets the gradient to 0, and initializes the previous values to an empty sequence.
  let output = Value(
    data: self.data / other.data,
    grad: 0.0,
    prev: @[self, other],
    op: "/"
  )
  output.backward_fn = proc() =
    self.grad += (1.0 / other.data) * output.grad
    other.grad -= (self.data / (other.data * other.data)) * output.grad
  return output

proc relu*(self: Value): Value =
  ## TLDR: ReLU activation function with gradient computation
  ## This is the ReLU activation function for the Value type. It creates a new Value object with the ReLU of the data of the value.
  ## It sets the gradient to 0, and initializes the previous values to an empty sequence.
  let output = Value(
    data: if self.data > 0: self.data else: 0,
    grad: 0.0,
    prev: @[self],
    op: "ReLU"
  )
  output.backward_fn = proc() =
    self.grad += (if self.data > 0: 1 else: 0) * output.grad
  return output

proc exp*(self: Value): Value =
  ## Exponential function for Value type
  let output = Value(
    data: exp(self.data),
    grad: 0.0,
    prev: @[self],
    op: "exp"
  )
  output.backward_fn = proc() =
    self.grad += output.data * output.grad
  return output

proc log*(self: Value): Value =
  ## Natural logarithm function for Value type
  let output = Value(
    data: ln(self.data),  # ln is from std/math
    grad: 0.0,
    prev: @[self],
    op: "log"
  )
  output.backward_fn = proc() =
    self.grad += (1.0 / self.data) * output.grad
  return output

proc backward*(self: Value) =
  ## TLDR: Backward pass for gradient computation
  ## This is the backward pass for the Value type. It computes the gradient of the output value with respect to the input values.
  ## It creates a topological order of the values and resets the gradients of all values.
  ## It then sets the gradient of the output value to 1 and processes the values in reverse order to compute the gradients.
  var topo: seq[Value] = @[]
  var visited: HashSet[Value] = initHashSet[Value]()
  
  proc buildTopo(v: Value) =
    if v notin visited:
      visited.incl(v)
      for child in v.prev:
        buildTopo(child)
      topo.add(v)
  buildTopo(self)
  
  # Reset gradients
  for node in topo:
    node.grad = 0.0
  
  # Set output gradient
  self.grad = 1.0
  
  # Process in reverse order
  for node in toSeq(topo).reversed():
    if node.backward_fn != nil:
      node.backward_fn()

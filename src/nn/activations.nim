import ../engine/value
import ./module
import std/sequtils

type
  ReLU* = ref object of Module

  Tanh* = ref object of Module

proc initReLU*(): ReLU =
  new(result)

proc initTanh*(): Tanh =
  new(result)

method forward*(self: ReLU, x: seq[Value]): seq[Value] =
  result = x.map(proc(v: Value): Value = v.relu())

method forward*(self: Tanh, x: seq[Value]): seq[Value] =
  result = x.map(proc(v: Value): Value =
    let e2x = (value(2.0) * v).exp()
    (e2x - value(1.0)) / (e2x + value(1.0))
  )
method forward*(self: ReLU, x: seq[seq[seq[seq[Value]]]]): seq[seq[seq[seq[Value]]]] =
  ## Forward pass for ReLU layer with 4D input (batch, channels, height, width)
  result = newSeqWith(x.len,
    newSeqWith(x[0].len,
      newSeqWith(x[0][0].len,
        newSeqWith(x[0][0][0].len,
          x[0][0][0][0].relu()
        )
      )
    )
  )
  
  # Apply ReLU to each element
  for b in 0..<x.len:
    for c in 0..<x[0].len:
      for h in 0..<x[0][0].len:
        for w in 0..<x[0][0][0].len:
          result[b][c][h][w] = x[b][c][h][w].relu()
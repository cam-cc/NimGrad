import std/[strformat]
import ../src/engine/value

proc `$`(v: Value): string =
  fmt"Value(data={v.data:.4f}, grad={v.grad:.4f})"

proc example1() =
  echo "\nExample 1: Basic operations and gradients"
  echo "----------------------------------------"
  let
    a = value(2.0)
    b = value(-3.0)
    c = value(10.0)
    
  let d = a * b + c
  d.backward()
  
  echo "Let d = a * b + c"
  echo fmt"a = {a}"
  echo fmt"b = {b}"
  echo fmt"c = {c}"
  echo fmt"d = {d}"

proc example2() =
  echo "\nExample 2: More complex expression"
  echo "----------------------------------------"
  let
    x = value(-4.0)
    y = value(2.0)
  
  let z = x * y + value(2.0)
  let q = z.relu() + value(3.0)
  q.backward()
  
  echo "Let z = x * y + 2"
  echo "Let q = relu(z) + 3"
  echo fmt"x = {x}"
  echo fmt"y = {y}"
  echo fmt"z = {z}"
  echo fmt"q = {q}"

proc example3() =
  echo "\nExample 3: Chain of operations"
  echo "----------------------------------------"
  let
    x = value(1.5)
    y = value(2.0)
  
  # Create a more complex computation graph
  let
    a = x * y    # 3.0
    b = a + x    # 4.5
    c = b.relu() # 4.5
    d = c * y    # 9.0
  
  d.backward()
  
  echo "Let a = x * y"
  echo "Let b = a + x"
  echo "Let c = relu(b)"
  echo "Let d = c * y"
  echo fmt"x = {x}"
  echo fmt"y = {y}"
  echo fmt"a = {a}"
  echo fmt"b = {b}"
  echo fmt"c = {c}"
  echo fmt"d = {d}"

when isMainModule:
  example1()
  example2()
  example3()
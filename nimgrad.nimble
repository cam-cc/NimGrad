# Package
version       = "0.1.0"
author        = "Cameron Coenjarts"
description   = "A lightweight autograd and neural network framework inspired by micrograd"
license       = "MIT"
srcDir        = "src"

# Dependencies
requires "nim >= 1.6.0"

# Tasks
task test, "Run the test suite":
  exec "testament pattern \"tests/*.nim\""

task examples, "Run examples":
  exec "nim c -r examples/scalar_autograd.nim"
  exec "nim c -r examples/mlp.nim"
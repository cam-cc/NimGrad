import unittest
import std/[sequtils, random]
import ../src/engine/value
import ../src/nn/[module, conv2d]

suite "Conv2d Layer":
  test "basic forward pass":
    # Create a simple 1x1x3x3 input (batch_size=1, channels=1, height=3, width=3)
    let input = @[
      @[  # Batch size 1
        @[  # 1 input channel
          @[value(1.0), value(2.0), value(3.0)],
          @[value(4.0), value(5.0), value(6.0)],
          @[value(7.0), value(8.0), value(9.0)]
        ]
      ]
    ]
    
    # Create Conv2d layer with 1 input channel, 1 output channel, and 2x2 kernel
    let conv = initConv2d(
      inChannels = 1,
      outChannels = 1,
      kernelSize = (height: 2, width: 2)
    )
    
    # Set weights manually for testing
    conv.weights[0][0][0][0] = value(1.0)  # Top-left
    conv.weights[0][0][0][1] = value(1.0)  # Top-right
    conv.weights[0][0][1][0] = value(1.0)  # Bottom-left
    conv.weights[0][0][1][1] = value(1.0)  # Bottom-right
    conv.bias[0] = value(0.0)  # Set bias to 0
    
    # Forward pass
    let output = conv.forward(input)
    
    # Expected shape: [1, 1, 2, 2] (batch_size=1, out_channels=1, height=2, width=2)
    check output.len == 1  # Batch size
    check output[0].len == 1  # Output channels
    check output[0][0].len == 2  # Height
    check output[0][0][0].len == 2  # Width
    
    # First output value should be sum of 2x2 window: 1+2+4+5 = 12
    check output[0][0][0][0].data == 12.0
    
    # Test backprop
    output[0][0][0][0].backward()
    
    # Check if gradients are propagated
    check conv.weights[0][0][0][0].grad != 0.0
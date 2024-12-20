import std/[sequtils, math, random]
import ../engine/value
import ./module

type
  Conv2d* = ref object of Module
    inChannels*: int
    outChannels*: int
    kernelSize*: tuple[height, width: int]
    stride*: tuple[height, width: int]
    padding*: tuple[height, width: int]
    weights*: seq[seq[seq[seq[Value]]]]  # [outChannels][inChannels][kernelHeight][kernelWidth]
    bias*: seq[Value]

proc initConv2d*(inChannels, outChannels: int, 
                kernelSize: tuple[height, width: int],
                stride: tuple[height, width: int] = (1, 1),
                padding: tuple[height, width: int] = (0, 0)): Conv2d =
  ## Initialize a new Conv2d layer
  result = Conv2d(
    inChannels: inChannels,
    outChannels: outChannels,
    kernelSize: kernelSize,
    stride: stride,
    padding: padding
  )
  
  # Initialize weights with Kaiming/He initialization
  randomize()
  let fanIn = inChannels * kernelSize.height * kernelSize.width
  let scale = sqrt(2.0 / fanIn.float)
  
  # Create 4D weight tensor
  result.weights = newSeqWith(outChannels,
    newSeqWith(inChannels,
      newSeqWith(kernelSize.height,
        newSeqWith(kernelSize.width,
          value(rand(-1.0..1.0) * scale)
        )
      )
    )
  )
  
  # Initialize biases to zero
  result.bias = newSeqWith(outChannels, value(0.0))
  
  # Add parameters to the base class
  var params: seq[Value] = @[]
  # Flatten weights for parameter collection
  for oc in result.weights:
    for ic in oc:
      for row in ic:
        params.add(row)
  params.add(result.bias)
  result.parameters = params

proc pad(input: seq[seq[seq[Value]]], padding: tuple[height, width: int]): seq[seq[seq[Value]]] =
  ## Add padding to the input tensor
  let
    channels = input.len
    height = input[0].len
    width = input[0][0].len
    padHeight = padding.height * 2
    padWidth = padding.width * 2
  
  result = newSeqWith(channels,
    newSeqWith(height + padHeight,
      newSeqWith(width + padWidth,
        value(0.0)
      )
    )
  )
  
  # Copy input data to padded tensor
  for c in 0..<channels:
    for h in 0..<height:
      for w in 0..<width:
        result[c][h + padding.height][w + padding.width] = input[c][h][w]

method forward*(self: Conv2d, input: seq[seq[seq[seq[Value]]]]): seq[seq[seq[seq[Value]]]] =
  echo "Input to Conv2d: ", input.len, "x", input[0].len, "x", input[0][0].len, "x", input[0][0][0].len
  let
    batchSize = input.len
    inHeight = input[0][0].len
    inWidth = input[0][0][0].len
    
    # Calculate output dimensions
    outHeight = ((inHeight + 2 * self.padding.height - self.kernelSize.height) div self.stride.height) + 1
    outWidth = ((inWidth + 2 * self.padding.width - self.kernelSize.width) div self.stride.width) + 1
  
  echo "Output dimensions: ", batchSize, "x", self.outChannels, "x", outHeight, "x", outWidth
  
  # Add padding if needed
  let padded: seq[seq[seq[seq[Value]]]] = 
    if self.padding.height > 0 or self.padding.width > 0:
      var paddedBatches = newSeq[seq[seq[seq[Value]]]](batchSize)
      for b in 0..<batchSize:
        paddedBatches[b] = pad(input[b], self.padding)
      paddedBatches
    else:
      input

  # Initialize output tensor
  result = newSeqWith(batchSize,
    newSeqWith(self.outChannels,
      newSeqWith(outHeight,
        newSeqWith(outWidth,
          value(0.0)
        )
      )
    )
  )
  
  # Perform convolution
  for b in 0..<batchSize:
    for oc in 0..<self.outChannels:
      for h in 0..<outHeight:
        for w in 0..<outWidth:
          var sum = self.bias[oc]
          
          # Convolve at this position
          for ic in 0..<self.inChannels:
            for kh in 0..<self.kernelSize.height:
              for kw in 0..<self.kernelSize.width:
                let
                  inH = h * self.stride.height + kh
                  inW = w * self.stride.width + kw
                
                # Check if inH and inW are within bounds
                if inH < padded[b][ic].len and inW < padded[b][ic][inH].len:
                  sum = sum + (padded[b][ic][inH][inW] * self.weights[oc][ic][kh][kw])
          
          result[b][oc][h][w] = sum
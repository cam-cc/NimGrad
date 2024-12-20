import std/[strformat, sequtils, random]
import ../src/engine/value
import ../src/nn/[module, conv2d, linear, activations, loss]
import ../src/tensor/tensor_ops

type
  MnistCNN* = ref object of Module
    conv1*: Conv2d
    relu1*: ReLU
    conv2*: Conv2d
    relu2*: ReLU
    fc1*: Linear
    relu3*: ReLU
    fc2*: Linear
    criterion*: CrossEntropyLoss

proc initMnistCNN*(): MnistCNN =
  result = MnistCNN()
  
  result.conv1 = initConv2d(
    inChannels = 1,
    outChannels = 16,
    kernelSize = (height: 3, width: 3),
    padding = (height: 1, width: 1)
  )
  result.relu1 = initReLU()
  
  result.conv2 = initConv2d(
    inChannels = 16,
    outChannels = 32,
    kernelSize = (height: 3, width: 3),
    stride = (height: 2, width: 2)
  )
  result.relu2 = initReLU()
  
  result.fc1 = initLinear(
    inFeatures = 32 * 13 * 13,
    outFeatures = 128
  )
  result.relu3 = initReLU()
  
  result.fc2 = initLinear(
    inFeatures = 128,
    outFeatures = 10
  )
  
  result.criterion = initCrossEntropyLoss()
  
  result.parameters = concat(
    result.conv1.parameters,
    result.conv2.parameters,
    result.fc1.parameters,
    result.fc2.parameters
  )

method forward*(self: MnistCNN, x: seq[seq[seq[seq[Value]]]]): seq[seq[seq[seq[Value]]]] =
  var features = self.conv1.forward(x)
  echo "After conv1: ", features.len, "x", features[0].len, "x", features[0][0].len, "x", features[0][0][0].len
  features = self.relu1.forward(features)
  features = self.conv2.forward(features)
  echo "After conv2: ", features.len, "x", features[0].len, "x", features[0][0].len, "x", features[0][0][0].len
  features = self.relu2.forward(features)
  
  let flattened = features.concat().concat().concat()
  echo "Flattened length: ", flattened.len
  
  # FC layers
  var fcOut = self.fc1.forward(flattened)
  fcOut = self.relu3.forward(fcOut)
  let logits = self.fc2.forward(fcOut)
  
  # Reshape logits back to 4D format [batch, classes, 1, 1]
  result = newSeqWith(logits.len,
    newSeqWith(10,  # number of classes
      newSeqWith(1,
        newSeqWith(1,
          value(0.0)
        )
      )
    )
  )
  
  for i in 0..<logits.len:
    for j in 0..<10:
      result[i][j][0][0] = logits[j]

proc train*(self: MnistCNN, trainX: seq[seq[seq[seq[Value]]]], trainY: seq[int], 
            epochs: int = 10, learningRate: float = 0.01, batchSize: int = 32) =
  let numSamples = trainX.len
  let numBatches = numSamples div batchSize
  
  for epoch in 0 ..< epochs:
    var epochLoss = value(0.0)
    var correctPreds = 0
    
    for batch in 0 ..< numBatches:
      let startIdx = batch * batchSize
      let endIdx = min(startIdx + batchSize, numSamples)
      
      let batchX = trainX[startIdx ..< endIdx]
      let batchY = trainY[startIdx ..< endIdx]
      
      # Forward pass
      let output = self.forward(batchX)
      
      # Calculate loss for each sample in batch
      var batchLoss = value(0.0)
      for i in 0 ..< batchY.len:
        # Extract logits for current sample
        var sampleLogits = newSeq[Value](10)
        for j in 0..<10:
          sampleLogits[j] = output[i][j][0][0]
        
        # Calculate loss
        let loss = self.criterion.forward(sampleLogits, batchY[i])
        batchLoss = batchLoss + loss
        
        # Track accuracy
        var maxIdx = 0
        var maxVal = sampleLogits[0].data
        for j in 1..<10:
          if sampleLogits[j].data > maxVal:
            maxVal = sampleLogits[j].data
            maxIdx = j
        if maxIdx == batchY[i]:
          correctPreds += 1
      
      # Backward pass
      self.zeroGrad()
      batchLoss.backward()
      
      # Update parameters
      for p in self.parameters:
        p.data -= learningRate * p.grad
      
      epochLoss = epochLoss + batchLoss
    
    let accuracy = correctPreds.float / numSamples.float * 100.0
    echo fmt"Epoch {epoch}: loss = {epochLoss.data / numSamples.float:.4f}, accuracy = {accuracy:.2f}%"

when isMainModule:
  randomize(42)
  
  let model = initMnistCNN()
  
  # Create dummy training data (100 samples)
  let x = newSeqWith(100,  # batch size
    newSeqWith(1,    # channels
      newSeqWith(28, # height
        newSeqWith(28, # width
          value(rand(0.0..1.0))
        )
      )
    )
  )
  
  var y = newSeq[int](100)
  for i in 0 ..< 100:
    y[i] = rand(0..9)
  
  echo "Training started..."
  model.train(x, y, epochs=5, batchSize=32)
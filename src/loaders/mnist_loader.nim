import std/[streams, endians, sequtils, random]
import ../engine/value

type
  MnistData* = object
    images*: seq[seq[seq[seq[Value]]]]  # [batch][channel][height][width]
    labels*: seq[seq[Value]]            # [batch][class]

proc readInt32BE(s: Stream): int =
  var val: int32
  discard s.readData(addr(val), sizeof(int32))
  bigEndian32(addr(result), addr(val))

proc loadMNIST*(imagesPath, labelsPath: string, batchSize: int = 32): MnistData =
  # Read images
  var f = newFileStream(imagesPath, fmRead)
  if f == nil:
    raise newException(IOError, "Cannot open images file")
  
  discard f.readInt32BE() # magic number
  let numImages = f.readInt32BE()
  let numRows = f.readInt32BE()
  let numCols = f.readInt32BE()
  
  # Read labels
  var l = newFileStream(labelsPath, fmRead)
  if l == nil:
    raise newException(IOError, "Cannot open labels file")
  
  discard l.readInt32BE() # magic number
  let numLabels = l.readInt32BE()
  
  assert numImages == numLabels
  
  # Read data
  result.images = @[]
  result.labels = @[]
  
  var currentBatch: seq[seq[seq[seq[Value]]]] = @[]
  var currentLabels: seq[seq[Value]] = @[]
  
  for i in 0..<numImages:
    # Read image
    var img = newSeqWith(1, # channels
      newSeqWith(numRows,
        newSeqWith(numCols, value(0.0))
      )
    )
    
    for r in 0..<numRows:
      for c in 0..<numCols:
        let pixel = float(f.readChar().uint8) / 255.0
        img[0][r][c] = value(pixel)
    
    # Read label
    let label = int(l.readChar().uint8)
    var oneHot = newSeq[Value](10)
    for j in 0..<10:
      oneHot[j] = value(if j == label: 1.0 else: 0.0)
    
    currentBatch.add(@[img])  # Add single-channel image
    currentLabels.add(oneHot)
    
    # If batch is complete or this is the last image
    if currentBatch.len == batchSize or i == numImages - 1:
      result.images.add(currentBatch)
      result.labels.add(currentLabels)
      currentBatch = @[]
      currentLabels = @[]
  
  f.close()
  l.close()

proc shuffleMNIST*(data: var MnistData) =
  ## Randomly shuffle the batches
  var indices = toSeq(0..<data.images.len)
  shuffle(indices)
  
  let oldImages = data.images
  let oldLabels = data.labels
  
  for i in 0..<indices.len:
    data.images[i] = oldImages[indices[i]]
    data.labels[i] = oldLabels[indices[i]]
import ../engine/value

proc reshape*(x: seq[seq[seq[seq[Value]]]]): seq[Value] =
  ## Reshapes a 4D tensor into a 2D tensor (flattens all dimensions except batch)
  result = @[]
  for batch in x:
    var batchFlat: seq[Value] = @[]
    for channel in batch:
      for row in channel:
        batchFlat.add(row)
    result.add(batchFlat)
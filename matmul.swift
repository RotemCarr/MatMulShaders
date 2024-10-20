import Metal

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device.")
}

let currentPath = FileManager.default.currentDirectoryPath
let filePath = "\(currentPath)/MatrixMultiply.metal"
guard let shaderSource = try? String(contentsOfFile: filePath, encoding: .utf8) else {
    fatalError("Failed to load shader file.")
}

guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else {
    fatalError("Failed to create Metal library.")
}

guard let function = library.makeFunction(name: "matrix_multiply") else {
    fatalError("Failed to create Metal function.")
}

guard let pipelineState = try? device.makeComputePipelineState(function: function) else {
    fatalError("Failed to create compute pipeline state.")
}

guard let commandQueue = device.makeCommandQueue() else {
    fatalError("Failed to create command queue.")
}

let widthA = 3
let heightA = 3
let widthB = 3
let heightB = 3

let matrixA: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
let matrixB: [Float] = [9, 8, 7, 6, 5, 4, 3, 2, 1]

var result = [Float](repeating: 0, count: widthA * heightB)

guard let bufferA = device.makeBuffer(bytes: matrixA, length: matrixA.count * MemoryLayout<Float>.size, options: .storageModeShared) else {
    fatalError("Failed to create buffer for matrix A")
}

guard let bufferB = device.makeBuffer(bytes: matrixB, length: matrixB.count * MemoryLayout<Float>.size, options: .storageModeShared) else {
    fatalError("Failed to create buffer for matrix B")
}

guard let bufferResult = device.makeBuffer(bytes: &result, length: result.count * MemoryLayout<Float>.size, options: .storageModeShared) else {
    fatalError("Failed to create buffer for the result matrix")
}

guard let bufferWidthA = device.makeBuffer(bytes: [UInt32(widthA)], length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
    fatalError("Failed to create buffer for widthA")
}

guard let bufferWidthB = device.makeBuffer(bytes: [UInt32(widthB)], length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
    fatalError("Failed to create buffer for widthB")
}

guard let commandBuffer = commandQueue.makeCommandBuffer() else {
    fatalError("Failed to create command buffer.")
}

guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
    fatalError("Failed to create compute command encoder.")
}

computeEncoder.setComputePipelineState(pipelineState)
computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
computeEncoder.setBuffer(bufferB, offset: 0, index: 1)
computeEncoder.setBuffer(bufferResult, offset: 0, index: 2)
computeEncoder.setBuffer(bufferWidthA, offset: 0, index: 3)
computeEncoder.setBuffer(bufferWidthB, offset: 0, index: 4)

let gridSize = MTLSize(width: widthB, height: heightA, depth: 1)
let threadGroupSize = MTLSize(width: 1, height: 1, depth: 1)
computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)

computeEncoder.endEncoding()

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

let resultPointer = bufferResult.contents().bindMemory(to: Float.self, capacity: result.count)
let output = Array(UnsafeBufferPointer(start: resultPointer, count: result.count))

print("Result Matrix:")
for row in 0..<heightA {
    print(output[row * widthB..<(row + 1) * widthB])
}

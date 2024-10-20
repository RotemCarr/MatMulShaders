import Foundation
import Metal

func parseCommandLineArguments() -> ([Float], Int, Int, [Float], Int, Int)? {
    let args = CommandLine.arguments
    guard args.count > 1 else {
        print("Usage: MatrixMultiply widthA heightA widthB heightB matrixA_elements matrixB_elements")
        return nil
    }

    // Parse matrix dimensions
    guard let widthA = Int(args[1]), let heightA = Int(args[2]),
          let widthB = Int(args[3]), let heightB = Int(args[4]) else {
        print("Invalid matrix dimensions provided.")
        return nil
    }

    // Ensure matrices are compatible for multiplication
    guard widthA == heightB else {
        print("Matrix dimensions are not compatible for multiplication. widthA must equal heightB.")
        return nil
    }

    // Parse matrix elements
    let matrixAElementsCount = widthA * heightA
    let matrixBElementsCount = widthB * heightB
    let expectedArgumentCount = 5 + matrixAElementsCount + matrixBElementsCount
    guard args.count == expectedArgumentCount else {
        print("Expected \(matrixAElementsCount + matrixBElementsCount) matrix elements, but received \(args.count - 5).")
        return nil
    }

    // Extract matrix elements
    let matrixA = args[5..<(5 + matrixAElementsCount)].compactMap { Float($0) }
    let matrixB = args[(5 + matrixAElementsCount)..<expectedArgumentCount].compactMap { Float($0) }

    // Ensure all elements were parsed correctly
    guard matrixA.count == matrixAElementsCount, matrixB.count == matrixBElementsCount else {
        print("Failed to parse matrix elements.")
        return nil
    }

    return (matrixA, widthA, heightA, matrixB, widthB, heightB)
}

func metalMatrixMultiply(matrixA: [Float], widthA: Int, heightA: Int,
                         matrixB: [Float], widthB: Int, heightB: Int) -> [Float]? {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Metal is not supported on this device.")
        return nil
    }
    let currentPath = FileManager.default.currentDirectoryPath
    let filePath = "\(currentPath)/MatrixMultiply.metal"
    guard let shaderSource = try? String(contentsOfFile: filePath, encoding: .utf8) else {
        print("Failed to load shader file.")
        return nil
    }

    guard let library = try? device.makeLibrary(source: shaderSource, options: nil),
          let function = library.makeFunction(name: "matrix_multiply"),
          let pipelineState = try? device.makeComputePipelineState(function: function),
          let commandQueue = device.makeCommandQueue() else {
        print("Failed to set up Metal.")
        return nil
    }

    var result = [Float](repeating: 0, count: widthA * heightB)

    guard let bufferA = device.makeBuffer(bytes: matrixA, length: matrixA.count * MemoryLayout<Float>.size, options: []),
          let bufferB = device.makeBuffer(bytes: matrixB, length: matrixB.count * MemoryLayout<Float>.size, options: []),
          let bufferResult = device.makeBuffer(bytes: &result, length: result.count * MemoryLayout<Float>.size, options: []),
          let bufferWidthA = device.makeBuffer(bytes: [UInt32(widthA)], length: MemoryLayout<UInt32>.size, options: []),
          let bufferWidthB = device.makeBuffer(bytes: [UInt32(widthB)], length: MemoryLayout<UInt32>.size, options: []) else {
        print("Failed to create Metal buffers.")
        return nil
    }

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
        print("Failed to create command buffer or encoder.")
        return nil
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

    return output
}

// Main execution
if let (matrixA, widthA, heightA, matrixB, widthB, heightB) = parseCommandLineArguments() {
    if let result = metalMatrixMultiply(matrixA: matrixA, widthA: widthA, heightA: heightA, matrixB: matrixB, widthB: widthB, heightB: heightB) {
        print("Result Matrix:")
        for row in 0..<heightA {
            print(result[row * widthB..<(row + 1) * widthB])
        }
    } else {
        print("Matrix multiplication failed.")
    }
} else {
    print("Failed to parse command-line arguments.")
}
#include <metal_stdlib>
using namespace metal;

kernel void matrix_multiply(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    constant uint& widthA [[buffer(3)]],
    constant uint& widthB [[buffer(4)]])
{
    // Get the row and column index for this thread
    uint row = gid.y;
    uint col = gid.x;
    
    float sum = 0.0;
    for (uint k = 0; k < widthA; ++k) {
        sum += matrixA[row * widthA + k] * matrixB[k * widthB + col];
    }
    
    result[row * widthB + col] = sum;
}

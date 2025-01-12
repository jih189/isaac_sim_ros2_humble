#include "multiply.h"
#include <cuda_runtime.h>

// CUDA kernel to multiply each element in the array by 2
__global__ void multiplyByTwoKernel(float* d_array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_array[idx] *= 2.0f;
    }
}

// Host function to launch the kernel
void multiplyByTwo(float* d_array, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    multiplyByTwoKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, size);
    cudaDeviceSynchronize();
}
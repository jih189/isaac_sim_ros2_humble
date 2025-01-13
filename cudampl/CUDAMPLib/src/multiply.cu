#include "multiply.h"
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

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

void test_multiplyByTwo() {
    // Initialize a vector with some elements
    std::vector<float> h_array = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int size = h_array.size();
    int bytes = size * sizeof(float);

    // Allocate device memory
    float* d_array;
    cudaMalloc(&d_array, bytes);

    // Copy data from host to device
    cudaMemcpy(d_array, h_array.data(), bytes, cudaMemcpyHostToDevice);

    // Call the function to multiply by two
    multiplyByTwo(d_array, size);

    // Copy the results back to host
    cudaMemcpy(h_array.data(), d_array, bytes, cudaMemcpyDeviceToHost);

    // Print the updated vector
    std::cout << "Updated vector: ";
    for (const auto& num : h_array) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_array);
}
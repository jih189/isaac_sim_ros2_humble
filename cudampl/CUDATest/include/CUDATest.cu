#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include "multiply.h"
#include <iostream>
#include <vector>

TEST(Test, lib)
{
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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
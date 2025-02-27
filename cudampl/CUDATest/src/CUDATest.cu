#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

// // Simple float3 struct. You can also use CUDA's float3 type.
// struct float3 {
//     float x, y, z;
// };

// CPU helper: compute Euclidean distance between two points.
float distance(const float3& a, const float3& b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Device helper: compute distance.
__device__ float deviceDistance(const float3& a, const float3& b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

// Device helper: linearly interpolate between two points.
__device__ float3 deviceInterpolate(const float3& a, const float3& b, float t) {
    float3 result;
    result.x = a.x + (b.x - a.x) * t;
    result.y = a.y + (b.y - a.y) * t;
    result.z = a.z + (b.z - a.z) * t;
    return result;
}

/*
 * Kernel: for each pair of points (p0 and p1), compute the interpolated points.
 * - set1, set2: arrays of endpoints (length n).
 * - stepSize: the spatial distance between successive interpolated points.
 * - output: a flattened 2D array of size n * maxSteps. Each row corresponds to one interpolation path.
 * - numSteps: array of length n; for each pair, the actual number of points computed.
 * - maxSteps: maximum number of steps (per path) allocated in the output.
 * - n: number of point pairs.
 */
__global__ void interpolateKernel(const float3* set1, const float3* set2, float stepSize,
                                  float3* output, int* numSteps, int maxSteps, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float3 p0 = set1[idx];
        float3 p1 = set2[idx];
        float d = deviceDistance(p0, p1);

        // Compute how many points: at least 2 (start and end).
        int steps = static_cast<int>(ceilf(d / stepSize)) + 1;
        if (steps < 2) steps = 2;
        numSteps[idx] = steps;

        // Fill the path for this pair.
        // For indices 0 to steps-2, compute t normally.
        for (int i = 0; i < steps - 1; ++i) {
            // Avoid division by zero. t goes from 0 to (nearly) 1.
            float t = (d > 1e-6f) ? (i * stepSize / d) : 0.0f;
            if (t > 1.0f)
                t = 1.0f;
            output[idx * maxSteps + i] = deviceInterpolate(p0, p1, t);
        }
        // Ensure the last point is exactly the endpoint.
        output[idx * maxSteps + steps - 1] = p1;
        // (Optional) For indices beyond steps, fill with the endpoint.
        for (int i = steps; i < maxSteps; ++i) {
            output[idx * maxSteps + i] = p1;
        }
    }
}

int main() {
    // Example data: two sets of two points each.
    std::vector<float3> h_set1 = { {0, 0, 0}, {1, 1, 1} };
    std::vector<float3> h_set2 = { {10, 10, 10}, {20, 20, 20} };
    int n = h_set1.size();

    // Spatial step size.
    float stepSize = 3.0f; // distance between successive interpolated points

    // Compute maximum number of steps required over all pairs (on host).
    int maxSteps = 0;
    std::vector<int> h_numSteps(n);
    for (int i = 0; i < n; ++i) {
        float d = distance(h_set1[i], h_set2[i]);
        int steps = static_cast<int>(ceil(d / stepSize)) + 1;
        if (steps < 2) steps = 2;
        h_numSteps[i] = steps;
        if (steps > maxSteps) {
            maxSteps = steps;
        }
    }

    std::cout << "Max steps per path: " << maxSteps << "\n";

    // Allocate device memory.
    float3 *d_set1, *d_set2, *d_output;
    int *d_numSteps;
    cudaMalloc(&d_set1, n * sizeof(float3));
    cudaMalloc(&d_set2, n * sizeof(float3));
    cudaMalloc(&d_output, n * maxSteps * sizeof(float3));
    cudaMalloc(&d_numSteps, n * sizeof(int));

    // Copy input data to device.
    cudaMemcpy(d_set1, h_set1.data(), n * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_set2, h_set2.data(), n * sizeof(float3), cudaMemcpyHostToDevice);

    // Launch kernel: one thread per point pair.
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    interpolateKernel<<<blocks, threadsPerBlock>>>(d_set1, d_set2, stepSize, d_output, d_numSteps, maxSteps, n);
    cudaDeviceSynchronize();

    // Copy results back to host.
    std::vector<float3> h_output(n * maxSteps);
    std::vector<int> h_outputNumSteps(n);
    cudaMemcpy(h_output.data(), d_output, n * maxSteps * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputNumSteps.data(), d_numSteps, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print out the paths.
    for (int i = 0; i < n; ++i) {
        std::cout << "Path " << i << " (" << h_outputNumSteps[i] << " points):\n";
        for (int j = 0; j < h_outputNumSteps[i]; ++j) {
            float3 pt = h_output[i * maxSteps + j];
            std::cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")\n";
        }
        std::cout << std::endl;
    }

    // Cleanup device memory.
    cudaFree(d_set1);
    cudaFree(d_set2);
    cudaFree(d_output);
    cudaFree(d_numSteps);

    return 0;
}
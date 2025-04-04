#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <queue>
#include <utility>  // For std::pair
#include <iostream>
#include <cmath>
#include <algorithm>

// This is a macro to check CUDA errors.
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

namespace CUDAMPLib
{
    std::vector<float> floatVectorFlatten(const std::vector<std::vector<float>>& input);
    std::vector<float> IsometryVectorFlatten(const std::vector<Eigen::Isometry3d>& transforms);
    std::vector<float> Vector3dflatten(const std::vector<Eigen::Vector3d>& vectors);
    std::vector<int> boolMatrixFlatten(const std::vector<std::vector<bool>>& input);
    std::vector<int> boolVectorFlatten(const std::vector<bool>& input);
    std::vector<Eigen::Isometry3d> fromFloatVectorToIsometry3d(const std::vector<float>& data);
    std::vector<int> kLeastIndices(const std::vector<float>& nums, int k);
    std::vector<int> kLeastIndices(const std::vector<float>& nums, int k, const std::vector<int>& group_indices);
    std::vector<std::vector<float>> interpolateVectors(const std::vector<float>& v1, const std::vector<float>& v2, float resolution);
    std::vector<std::vector<float>> removeRedundantVectors(const std::vector<std::vector<float>>& vec, const float tol);
}
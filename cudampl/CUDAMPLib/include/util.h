#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <queue>
#include <utility>  // For std::pair
#include <iostream>


namespace CUDAMPLib
{
    std::vector<float> floatVectorFlatten(const std::vector<std::vector<float>>& input);
    std::vector<float> IsometryVectorFlatten(const std::vector<Eigen::Isometry3d>& transforms);
    std::vector<float> Vector3dflatten(const std::vector<Eigen::Vector3d>& vectors);
    std::vector<int> boolMatrixFlatten(const std::vector<std::vector<bool>>& input);
    std::vector<int> boolVectorFlatten(const std::vector<bool>& input);
    std::vector<Eigen::Isometry3d> fromFloatVectorToIsometry3d(const std::vector<float>& data);
    std::vector<int> kLeastIndices(const std::vector<float>& nums, int k);
}
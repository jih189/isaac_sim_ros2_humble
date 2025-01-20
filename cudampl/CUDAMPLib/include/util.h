#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>

#include <iostream>

namespace CUDAMPLib
{
    std::vector<float> floatVectorFlatten(const std::vector<std::vector<float>>& input);
    std::vector<float> IsometryVectorFlatten(const std::vector<Eigen::Isometry3d>& transforms);
    std::vector<std::vector<float>> FromFloatVectorToVec3(const std::vector<float>& data, size_t size);
    std::vector<Eigen::Isometry3d> fromFloatVector(const std::vector<float>& data);
    std::vector<float> Vector3dflatten(const std::vector<Eigen::Vector3d>& vectors);
}
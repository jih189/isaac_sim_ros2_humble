#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>

// kinematics forward function with C++ 
void kin_forward(
    const std::vector<std::vector<float>>& joint_values,
    const std::vector<int>& joint_types,
    const std::vector<Eigen::Isometry3d>& joint_poses,
    const std::vector<Eigen::Vector3d>& joint_axes,
    const std::vector<int>& link_maps,
    std::vector<std::vector<Eigen::Isometry3d>>& link_poses_set
);
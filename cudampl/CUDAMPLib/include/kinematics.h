#pragma once

#include "util.h"
#include "cost.h"

namespace CUDAMPLib
{
    #define CUDAMPLib_UNKNOWN 0
    #define CUDAMPLib_REVOLUTE 1
    #define CUDAMPLib_PRISMATIC 2
    #define CUDAMPLib_PLANAR 3
    #define CUDAMPLib_FLOATING 4
    #define CUDAMPLib_FIXED 5

    void fixed_joint_fn(
        const Eigen::Isometry3d& parent_link_pose, 
        const Eigen::Isometry3d& joint_pose, 
        Eigen::Isometry3d& link_pose
    );

    void prism_joint_fn(
        const Eigen::Isometry3d& parent_link_pose, 
        const Eigen::Isometry3d& joint_pose, 
        const Eigen::Vector3d& joint_axis, 
        const float joint_value, // joint value in meters
        Eigen::Isometry3d& link_pose
    );

    void revolute_joint_fn(
        const Eigen::Isometry3d& parent_link_pose, 
        const Eigen::Isometry3d& joint_pose, 
        const Eigen::Vector3d& joint_axis, 
        const float joint_value, // joint value in radians
        Eigen::Isometry3d& link_pose
    );

    // kinematics forward function with C++ 
    void kin_forward(
        const std::vector<std::vector<float>>& joint_values,
        const std::vector<int>& joint_types,
        const std::vector<Eigen::Isometry3d>& joint_poses,
        const std::vector<Eigen::Vector3d>& joint_axes,
        const std::vector<int>& link_maps,
        std::vector<std::vector<Eigen::Isometry3d>>& link_poses_set
    );

    void kin_forward_cuda(
        const std::vector<std::vector<float>>& joint_values,
        const std::vector<int>& joint_types,
        const std::vector<Eigen::Isometry3d>& joint_poses,
        const std::vector<Eigen::Vector3d>& joint_axes,
        const std::vector<int>& link_maps,
        std::vector<std::vector<Eigen::Isometry3d>>& link_poses_set
    );

    void kin_forward_collision_spheres_cuda(
        const std::vector<std::vector<float>>& joint_values,
        const std::vector<int>& joint_types,
        const std::vector<Eigen::Isometry3d>& joint_poses,
        const std::vector<Eigen::Vector3d>& joint_axes,
        const std::vector<int>& link_maps,
        const std::vector<int>& collision_spheres_map,
        const std::vector<std::vector<float>>& collision_spheres_pos,
        std::vector<std::vector<Eigen::Isometry3d>>& link_poses_set,
        std::vector<std::vector<std::vector<float>>>& collision_spheres_pos_in_baselink
    );

    void evaluation_cuda(
        const std::vector<std::vector<float>>& joint_values,
        const std::vector<int>& joint_types,
        const std::vector<Eigen::Isometry3d>& joint_poses,
        const std::vector<Eigen::Vector3d>& joint_axes,
        const std::vector<int>& link_maps,
        const std::vector<int>& collision_spheres_map,
        const std::vector<std::vector<float>>& collision_spheres_pos,
        const std::vector<float>& collision_spheres_radius,
        const std::vector<CostBasePtr>& costs,
        std::vector<float>& costs_values,
        std::vector<std::vector<std::vector<float>>>& collision_spheres_pos_in_baselink_for_debug
    );
}
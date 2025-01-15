#include "kinematics.h"
#include <iostream>

#define UNKNOWN 0
#define REVOLUTE 1
#define PRISMATIC 2
#define PLANAR 3 // This is not used
#define FLOATING 4 // This is not used
#define FIXED 5

void fixed_joint_fn(const Eigen::Isometry3d& parent_link_pose, const Eigen::Isometry3d& joint_pose, Eigen::Isometry3d& link_pose)
{
    link_pose = parent_link_pose * joint_pose;
}

void prism_joint_fn(
    const Eigen::Isometry3d& parent_link_pose, 
    const Eigen::Isometry3d& joint_pose, 
    const Eigen::Vector3d& joint_axis, 
    const float joint_value, // joint value in meters
    Eigen::Isometry3d& link_pose)
{
    Eigen::Isometry3d joint_transform = Eigen::Isometry3d::Identity();
    joint_transform.translation() = joint_axis * joint_value;
    link_pose = parent_link_pose * joint_pose * joint_transform;
}

void revolute_joint_fn(
    const Eigen::Isometry3d& parent_link_pose, 
    const Eigen::Isometry3d& joint_pose, 
    const Eigen::Vector3d& joint_axis, 
    const float joint_value, // joint value in radians
    Eigen::Isometry3d& link_pose)
{
    Eigen::Isometry3d joint_transform = Eigen::Isometry3d::Identity();
    joint_transform.rotate(Eigen::AngleAxisd(joint_value, joint_axis));
    link_pose = parent_link_pose * joint_pose * joint_transform;
}

void kin_forward(
    const std::vector<std::vector<float>>& joint_values,
    const std::vector<int>& joint_types,
    const std::vector<Eigen::Isometry3d>& joint_poses,
    const std::vector<Eigen::Vector3d>& joint_axes,
    const std::vector<int>& link_maps,
    std::vector<std::vector<Eigen::Isometry3d>>& link_poses_set)
{
    size_t num_links = link_maps.size();

    const std::vector<float> current_joint_values = joint_values[0];

    link_poses_set.resize(joint_values.size());

    // initialize link poses with size of link_maps
    std::vector<Eigen::Isometry3d> link_poses(num_links, Eigen::Isometry3d::Identity());

    // this j is index of active joint values
    size_t j = 0;
    // The first link is the base link, so we can skip it
    for(size_t i = 1; i < num_links; i++)
    {
        Eigen::Isometry3d parent_link_pose = link_poses[link_maps[i]];
        // Based on the joint type
        switch(joint_types[i])
        {
            case REVOLUTE:
                // std::cout << "Joint type: REVOLUTE" << std::endl;
                revolute_joint_fn(parent_link_pose, joint_poses[i], joint_axes[i], current_joint_values[j], link_poses[i]);
                j++;
                break;
            case PRISMATIC:
                // std::cout << "Joint type: PRISMATIC" << std::endl;
                prism_joint_fn(parent_link_pose, joint_poses[i], joint_axes[i], current_joint_values[j], link_poses[i]);
                j++;
                break;
            case FIXED:
                // std::cout << "Joint type: FIXED" << std::endl;
                fixed_joint_fn(parent_link_pose, joint_poses[i], link_poses[i]);
                break;
            default:
                // std::cout << "Unknown joint type" << std::endl;
                // Unknown joint type, throw an error
                throw std::runtime_error("Unknown joint type");
                break;
        }
    }

    link_poses_set[0] = link_poses;
}
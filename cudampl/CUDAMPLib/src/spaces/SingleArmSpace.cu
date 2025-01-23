#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014
#include <spaces/SingleArmSpace.h>
#include <cuda_runtime.h>

namespace CUDAMPLib {

    SingleArmSpace::SingleArmSpace(
        size_t dim,
        const std::vector<int>& joint_types,
        const std::vector<Eigen::Isometry3d>& joint_poses,
        const std::vector<Eigen::Vector3d>& joint_axes,
        const std::vector<int>& link_parent_link_maps,
        const std::vector<int>& collision_spheres_to_link_map,
        const std::vector<std::vector<float>>& collision_spheres_pos_in_link,
        const std::vector<float>& collision_spheres_radius,
        const std::vector<bool>& active_joint_map
    )
        : BaseSpace(dim)
    {
        // need to allocate device memory for joint_types, joint_poses, joint_axes, 
        // parent_link_maps, collision_spheres_to_link_map, collision_spheres_pos_in_link, 
        // and collision_spheres_radius
        num_of_joints = joint_types.size();
        num_of_links = link_parent_link_maps.size();
        num_of_self_collision_spheres = collision_spheres_to_link_map.size();

        int byte_size_of_pose_matrix = sizeof(float) * 4 * 4;
        int joint_types_bytes = sizeof(int) * num_of_joints;
        int joint_poses_bytes = byte_size_of_pose_matrix * num_of_joints;
        int joint_axes_bytes = sizeof(float) * 3 * num_of_joints;
        int link_parent_link_maps_bytes = sizeof(int) * num_of_links;
        int collision_spheres_to_link_map_bytes = sizeof(int) * num_of_self_collision_spheres;
        int collision_spheres_pos_in_link_bytes = sizeof(float) * 3 * num_of_self_collision_spheres;
        int collision_spheres_radius_bytes = sizeof(float) * num_of_self_collision_spheres;
        int active_joint_map_bytes = sizeof(int) * num_of_joints;
        
        // allocate device memory
        cudaMalloc(&d_joint_types, joint_types_bytes);
        cudaMalloc(&d_joint_poses, joint_poses_bytes);
        cudaMalloc(&d_joint_axes, joint_axes_bytes);
        cudaMalloc(&d_link_parent_link_maps, link_parent_link_maps_bytes);
        cudaMalloc(&d_collision_spheres_to_link_map, collision_spheres_to_link_map_bytes);
        cudaMalloc(&d_collision_spheres_pos_in_link, collision_spheres_pos_in_link_bytes);
        cudaMalloc(&d_collision_spheres_radius, collision_spheres_radius_bytes);
        cudaMalloc(&d_active_joint_map, active_joint_map_bytes);

        // copy data to device memory
        cudaMemcpy(d_joint_types, joint_types.data(), joint_types_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_joint_poses, IsometryVectorFlatten(joint_poses).data(), joint_poses_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_joint_axes, Vector3dflatten(joint_axes).data(), joint_axes_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_link_parent_link_maps, link_parent_link_maps.data(), link_parent_link_maps_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision_spheres_to_link_map, collision_spheres_to_link_map.data(), collision_spheres_to_link_map_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision_spheres_pos_in_link, floatVectorFlatten(collision_spheres_pos_in_link).data(), collision_spheres_pos_in_link_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision_spheres_radius, collision_spheres_radius.data(), collision_spheres_radius_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_active_joint_map, boolVectorFlatten(active_joint_map).data(), active_joint_map_bytes, cudaMemcpyHostToDevice);
    }

    SingleArmSpace::~SingleArmSpace()
    {
        // free device memory
        cudaFree(d_joint_types);
        cudaFree(d_joint_poses);
        cudaFree(d_joint_axes);
        cudaFree(d_link_parent_link_maps);
        cudaFree(d_collision_spheres_to_link_map);
        cudaFree(d_collision_spheres_pos_in_link);
        cudaFree(d_collision_spheres_radius);
        cudaFree(d_active_joint_map);
    }

    void SingleArmSpace::sample(int num_of_config, std::vector<std::vector<float>>& samples)
    {

    }

    void SingleArmSpace::getMotions(
        const std::vector<std::vector<float>>& start, 
        const std::vector<std::vector<float>>& end, 
        std::vector<std::vector<std::vector<float>>>& motions,
        std::vector<bool> motion_feasibility
    )
    {

    }

    void SingleArmSpace::checkMotions(
        const std::vector<std::vector<float>>& start, 
        const std::vector<std::vector<float>>& end, 
        std::vector<bool>& motion_feasibility
    )
    {

    }

    void SingleArmSpace::checkStates(
        const std::vector<std::vector<float>>& states,
        std::vector<bool>& state_feasibility
    )
    {

    }

} // namespace cudampl
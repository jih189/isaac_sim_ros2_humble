#include <spaces/SingleArmSpace.h>

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
        const std::vector<bool>& active_joint_map,
        const std::vector<float>& lower,
        const std::vector<float>& upper
    )
        : BaseSpace(dim)
    {
        // need to allocate device memory for joint_types, joint_poses, joint_axes, 
        // parent_link_maps, collision_spheres_to_link_map, collision_spheres_pos_in_link, 
        // and collision_spheres_radius
        num_of_joints = joint_types.size();
        num_of_links = link_parent_link_maps.size();
        num_of_self_collision_spheres = collision_spheres_to_link_map.size();

        // set bounds
        lower_bound = lower;
        upper_bound = upper;

        int byte_size_of_pose_matrix = sizeof(float) * 4 * 4;
        int joint_types_bytes = sizeof(int) * num_of_joints;
        int joint_poses_bytes = byte_size_of_pose_matrix * num_of_joints;
        int joint_axes_bytes = sizeof(float) * 3 * num_of_joints;
        int link_parent_link_maps_bytes = sizeof(int) * num_of_links;
        int collision_spheres_to_link_map_bytes = sizeof(int) * num_of_self_collision_spheres;
        int collision_spheres_pos_in_link_bytes = sizeof(float) * 3 * num_of_self_collision_spheres;
        int collision_spheres_radius_bytes = sizeof(float) * num_of_self_collision_spheres;
        int active_joint_map_bytes = sizeof(int) * num_of_joints;
        int lower_bound_bytes = sizeof(float) * dim;
        int upper_bound_bytes = sizeof(float) * dim;
        
        // allocate device memory
        cudaMalloc(&d_joint_types, joint_types_bytes);
        cudaMalloc(&d_joint_poses, joint_poses_bytes);
        cudaMalloc(&d_joint_axes, joint_axes_bytes);
        cudaMalloc(&d_link_parent_link_maps, link_parent_link_maps_bytes);
        cudaMalloc(&d_collision_spheres_to_link_map, collision_spheres_to_link_map_bytes);
        cudaMalloc(&d_collision_spheres_pos_in_link, collision_spheres_pos_in_link_bytes);
        cudaMalloc(&d_collision_spheres_radius, collision_spheres_radius_bytes);
        cudaMalloc(&d_active_joint_map, active_joint_map_bytes);
        cudaMalloc(&d_lower_bound, lower_bound_bytes);
        cudaMalloc(&d_upper_bound, upper_bound_bytes);

        // copy data to device memory
        cudaMemcpy(d_joint_types, joint_types.data(), joint_types_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_joint_poses, IsometryVectorFlatten(joint_poses).data(), joint_poses_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_joint_axes, Vector3dflatten(joint_axes).data(), joint_axes_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_link_parent_link_maps, link_parent_link_maps.data(), link_parent_link_maps_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision_spheres_to_link_map, collision_spheres_to_link_map.data(), collision_spheres_to_link_map_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision_spheres_pos_in_link, floatVectorFlatten(collision_spheres_pos_in_link).data(), collision_spheres_pos_in_link_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision_spheres_radius, collision_spheres_radius.data(), collision_spheres_radius_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_active_joint_map, boolVectorFlatten(active_joint_map).data(), active_joint_map_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_lower_bound, lower.data(), lower_bound_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_upper_bound, upper.data(), upper_bound_bytes, cudaMemcpyHostToDevice);
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
        cudaFree(d_lower_bound);
        cudaFree(d_upper_bound);
    }

    // __global__ void initCurand(curandState * state, unsigned long seed)
    // {
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //     curand_init(seed, idx, 0, &state[idx]);
    // }

    // __global__ void sample_kernel(
    //     float * d_sampled_states,
    //     int num_of_config,
    //     int num_of_joints,
    //     float * d_lower_bound,
    //     float * d_upper_bound
    // )
    // {
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (idx >= num_of_config) return;

    //     for (int i = 0; i < num_of_joints; i++)
    //     {
    //         d_sampled_states[idx * num_of_joints + i] = d_lower_bound[i] + (d_upper_bound[i] - d_lower_bound[i]) * (float)rand() / RAND_MAX;
    //     }
    // }

    BaseStatesPtr SingleArmSpace::sample(int num_of_config)
    {
        // Create a state
        SingleArmStatesPtr sampled_states = std::make_shared<SingleArmStates>(num_of_config, num_of_joints);

        // get device memory with size of num_of_config * num_of_joints * sizeof(float)
        float * d_sampled_states = sampled_states->getJointStatesCuda();

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_config + threadsPerBlock - 1) / threadsPerBlock;

        // // set random seed
        // unsigned long seed = 1234;
        // curandState *d_state;
        // cudaMalloc(&d_state, num_of_config * num_of_joints * sizeof(curandState));
        // initCurand<<<blocksPerGrid, threadsPerBlock>>>(d_state, seed);

        // // call kernel
        // sample_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_sampled_states, num_of_config, num_of_joints, d_lower_bound, d_upper_bound);

        // // wait for kernel to finish
        // cudaDeviceSynchronize();

        return sampled_states;
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
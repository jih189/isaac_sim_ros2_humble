#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include <spaces/SingleArmSpace.h>

#include <chrono>

namespace CUDAMPLib {

    SingleArmSpace::SingleArmSpace(
        size_t dim,
        const std::vector<BaseConstraintPtr>& constraints,
        const std::vector<int>& joint_types,
        const std::vector<Eigen::Isometry3d>& joint_poses,
        const std::vector<Eigen::Vector3d>& joint_axes,
        const std::vector<int>& link_parent_link_maps,
        const std::vector<int>& collision_spheres_to_link_map,
        const std::vector<std::vector<float>>& collision_spheres_pos_in_link,
        const std::vector<float>& collision_spheres_radius,
        const std::vector<bool>& active_joint_map,
        const std::vector<float>& lower,
        const std::vector<float>& upper,
        const std::vector<float>& default_joint_values,
        const std::vector<std::string>& link_names,
        float resolution
    )
        : BaseSpace(dim, constraints),
          gen(std::random_device{}()),
          resolution_(resolution),
          dist(0, std::numeric_limits<unsigned long>::max())
    {
        // need to allocate device memory for joint_types, joint_poses, joint_axes, 
        // parent_link_maps, collision_spheres_to_link_map, collision_spheres_pos_in_link, 
        // and collision_spheres_radius
        num_of_joints = joint_types.size();
        num_of_links = link_parent_link_maps.size();
        num_of_self_collision_spheres = collision_spheres_to_link_map.size();
        // copy data to member variables
        active_joint_map_ = active_joint_map;
        default_joint_values_ = default_joint_values;

        // set bounds
        lower_bound = lower;
        upper_bound = upper;

        // get the number of active joints
        num_of_active_joints_ = 0;
        for (size_t i = 0; i < active_joint_map.size(); i++)
        {
            if (active_joint_map[i])
            {
                num_of_active_joints_++;
            }
        }

        // set the link names
        link_names_ = link_names;

        if (num_of_links != link_names_.size())
        {
            throw std::runtime_error("Number of link names is not equal to the number of links");
        }

        int byte_size_of_pose_matrix = sizeof(float) * 4 * 4;
        int joint_types_bytes = sizeof(int) * num_of_joints;
        int joint_poses_bytes = byte_size_of_pose_matrix * num_of_joints;
        int joint_axes_bytes = sizeof(float) * 3 * num_of_joints;
        int link_parent_link_maps_bytes = sizeof(int) * num_of_links;
        int collision_spheres_to_link_map_bytes = sizeof(int) * num_of_self_collision_spheres;
        int self_collision_spheres_pos_in_link_bytes = sizeof(float) * 3 * num_of_self_collision_spheres;
        int self_collision_spheres_radius_bytes = sizeof(float) * num_of_self_collision_spheres;
        int active_joint_map_bytes = sizeof(int) * num_of_joints;
        int lower_bound_bytes = sizeof(float) * num_of_joints;
        int upper_bound_bytes = sizeof(float) * num_of_joints;
        int default_joint_values_bytes = sizeof(float) * num_of_joints;
        
        // allocate device memory
        cudaMalloc(&d_joint_types, joint_types_bytes);
        cudaMalloc(&d_joint_poses, joint_poses_bytes);
        cudaMalloc(&d_joint_axes, joint_axes_bytes);
        cudaMalloc(&d_link_parent_link_maps, link_parent_link_maps_bytes);
        cudaMalloc(&d_collision_spheres_to_link_map, collision_spheres_to_link_map_bytes);
        cudaMalloc(&d_self_collision_spheres_pos_in_link, self_collision_spheres_pos_in_link_bytes);
        cudaMalloc(&d_self_collision_spheres_radius, self_collision_spheres_radius_bytes);
        cudaMalloc(&d_active_joint_map, active_joint_map_bytes);
        cudaMalloc(&d_lower_bound, lower_bound_bytes);
        cudaMalloc(&d_upper_bound, upper_bound_bytes);
        cudaMalloc(&d_default_joint_values, default_joint_values_bytes);

        // copy data to device memory
        cudaMemcpy(d_joint_types, joint_types.data(), joint_types_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_joint_poses, IsometryVectorFlatten(joint_poses).data(), joint_poses_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_joint_axes, Vector3dflatten(joint_axes).data(), joint_axes_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_link_parent_link_maps, link_parent_link_maps.data(), link_parent_link_maps_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision_spheres_to_link_map, collision_spheres_to_link_map.data(), collision_spheres_to_link_map_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_self_collision_spheres_pos_in_link, floatVectorFlatten(collision_spheres_pos_in_link).data(), self_collision_spheres_pos_in_link_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_self_collision_spheres_radius, collision_spheres_radius.data(), self_collision_spheres_radius_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_active_joint_map, boolVectorFlatten(active_joint_map).data(), active_joint_map_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_lower_bound, lower.data(), lower_bound_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_upper_bound, upper.data(), upper_bound_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_default_joint_values, default_joint_values.data(), default_joint_values_bytes, cudaMemcpyHostToDevice);
    }

    SingleArmSpace::~SingleArmSpace()
    {
        // free device memory
        cudaFree(d_joint_types);
        cudaFree(d_joint_poses);
        cudaFree(d_joint_axes);
        cudaFree(d_link_parent_link_maps);
        cudaFree(d_collision_spheres_to_link_map);
        cudaFree(d_self_collision_spheres_pos_in_link);
        cudaFree(d_self_collision_spheres_radius);
        cudaFree(d_active_joint_map);
        cudaFree(d_lower_bound);
        cudaFree(d_upper_bound);
        cudaFree(d_default_joint_values);
    }

    __global__ void initCurand(curandState * state, unsigned long seed, int state_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= state_size) return;
        curand_init(seed, idx, 0, &state[idx]);
    }

    __global__ void sample_kernel(
        curandState_t * d_random_state,
        float * d_sampled_states,
        int num_of_config,
        int num_of_joints,
        int * d_active_joint_map,
        float * d_lower_bound,
        float * d_upper_bound,
        float * d_default_joint_values
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_config * num_of_joints) return;

        int joint_idx = idx % num_of_joints;

        // if joint is not active, then set the value to 0
        if (d_active_joint_map[joint_idx] == 0)
        {
            d_sampled_states[idx] = d_default_joint_values[joint_idx];
        }
        else
        {
            curandState_t local_state = d_random_state[idx];
            d_sampled_states[idx] = curand_uniform(&local_state) * (d_upper_bound[joint_idx] - d_lower_bound[joint_idx]) + d_lower_bound[joint_idx];
        }
    }

    __global__ void interpolate_kernel(
        float * d_from_states,
        float * d_to_states,
        int num_of_config,
        int num_of_joints,
        float * max_distance
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_config) return;

        float * from_state = &d_from_states[idx * num_of_joints];
        float * to_state = &d_to_states[idx * num_of_joints];

        // calculate the distance between the two states
        float distance = 0.0f;
        for (size_t i = 0; i < num_of_joints; i++)
        {
            distance += (from_state[i] - to_state[i]) * (from_state[i] - to_state[i]);
        }
        distance = sqrt(distance);

        if (distance > *max_distance)
        {
            // interpolate the two states
            for (size_t i = 0; i < num_of_joints; i++)
            {
                to_state[i] = from_state[i] + (*max_distance / distance) * (to_state[i] - from_state[i]);
            }
        }
    }

    BaseStatesPtr SingleArmSpace::sample(int num_of_config)
    {
        // first try to allocate d_random_state
        curandState * d_random_state;
        size_t d_random_state_bytes = num_of_config * num_of_joints * sizeof(curandState);
        auto allocate_result = cudaMalloc(&d_random_state, d_random_state_bytes);
        if (allocate_result != cudaSuccess)
        {
            // print in red
            std::cerr << "\033[31m" << "Failed to allocate device memory for random state. Perhaps, the num_of_config is too large." << "\033[0m" << std::endl;
            // fail to allocate memory
            return nullptr;
        }

        // get space info
        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        // Create a state
        SingleArmStatesPtr sampled_states = std::make_shared<SingleArmStates>(num_of_config, space_info);
        if(! sampled_states->isValid())
        {
            // free device memory
            cudaFree(d_random_state);

            sampled_states.reset();
            // fail to allocate memory
            return nullptr;
        }

        // get device memory with size of num_of_config * num_of_joints * sizeof(float)
        float * d_sampled_states = sampled_states->getJointStatesCuda();

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_config * num_of_joints + threadsPerBlock - 1) / threadsPerBlock;

        // // set random seed
        // std::random_device rd;  // Non-deterministic seed (preferred)
        // std::mt19937_64 gen(rd()); // 64-bit Mersenne Twister PRNG
        // std::uniform_int_distribution<unsigned long> dist(0, ULONG_MAX);

        unsigned long seed = dist(gen);
        initCurand<<<blocksPerGrid, threadsPerBlock>>>(d_random_state, seed, num_of_config * num_of_joints);

        // call kernel
        sample_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_random_state, 
            d_sampled_states, 
            num_of_config, 
            num_of_joints,
            d_active_joint_map,
            d_lower_bound, 
            d_upper_bound,
            d_default_joint_values
        );

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        // free device memory
        cudaFree(d_random_state);

        CUDA_CHECK(cudaGetLastError());

        if (projectable_constraint_indices_.size() > 0)
        {
            // This space has projectable constraints, so we need to project the sampled states first.
            this->projectStates(sampled_states);
        }

        return sampled_states;
    }

    BaseStatesPtr SingleArmSpace::createStatesFromVectorFull(const std::vector<std::vector<float>>& joint_values)
    {
        size_t num_of_config = joint_values.size();

        if (num_of_config == 0)
        {
            // throw an exception
            throw std::runtime_error("No joint values is empty");
        }

        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        // Create a state
        SingleArmStatesPtr generated_states = std::make_shared<SingleArmStates>(num_of_config, space_info);
        if(! generated_states->isValid())
        {
            generated_states.reset();
            // fail to allocate memory
            return nullptr;
        }

        // get device memory with size of num_of_config * num_of_joints * sizeof(float)
        float * d_generated_states = generated_states->getJointStatesCuda();

        // copy data to device memory
        cudaMemcpy(d_generated_states, floatVectorFlatten(joint_values).data(), num_of_config * num_of_joints * sizeof(float), cudaMemcpyHostToDevice);

        return generated_states;
    }

    BaseStatesPtr SingleArmSpace::createStatesFromVector(const std::vector<std::vector<float>>& joint_values)
    {
        int num_of_config = joint_values.size();

        if (num_of_config == 0)
        {
            // throw an exception
            throw std::runtime_error("No joint values is empty");
        }

        // check the size of the joint values is correct
        for (size_t i = 0; i < joint_values.size(); i++)
        {
            if (joint_values[i].size() != dim)
            {
                // throw an exception
                // throw std::runtime_error("Joint values size is not correct");
                throw std::runtime_error("Joint values size is not correct. Expected: " + std::to_string(dim) + " Got: " + std::to_string(joint_values[i].size()));
            }
        }

        // initialize the joint value with num_of_config * num_of_joints
        std::vector<std::vector<float>> joint_value_w_correct_size(num_of_config, std::vector<float>(num_of_joints, 0.0f));

        // copy the joint values to the correct size and set the default values for the inactive joints.
        for (size_t i = 0; i < joint_value_w_correct_size.size(); i++)
        {
            size_t k = 0;
            for (size_t j = 0; j < joint_value_w_correct_size[i].size(); j++)
            {
                if(active_joint_map_[j])
                {
                    joint_value_w_correct_size[i][j] = joint_values[i][k];
                    k++;
                }
                else
                {
                    joint_value_w_correct_size[i][j] = default_joint_values_[j];
                }
            }
        }

        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        // Create a state
        SingleArmStatesPtr generated_states = std::make_shared<SingleArmStates>(num_of_config, space_info);
        if(! generated_states->isValid())
        {
            generated_states.reset();
            // fail to allocate memory
            return nullptr;
        }

        // get device memory with size of num_of_config * num_of_joints * sizeof(float)
        float * d_generated_states = generated_states->getJointStatesCuda();

        // copy data to device memory
        cudaMemcpy(d_generated_states, floatVectorFlatten(joint_value_w_correct_size).data(), num_of_config * num_of_joints * sizeof(float), cudaMemcpyHostToDevice);

        return generated_states;
    }

    std::vector<std::vector<float>> SingleArmSpace::getJointVectorInActiveJointsFromStates(const BaseStatesPtr & states)
    {
        // static cast to SingleArmStatesPtr
        SingleArmStatesPtr single_arm_states = std::dynamic_pointer_cast<SingleArmStates>(states);

        // get the joint states from the states
        std::vector<std::vector<float>> joint_states = single_arm_states->getJointStatesFullHost();

        // filer the joint states with active_joint_map_
        std::vector<std::vector<float>> joint_states_filtered;
        for (size_t i = 0; i < joint_states.size(); i++)
        {
            std::vector<float> joint_state_filtered;
            for (size_t j = 0; j < joint_states[i].size(); j++)
            {
                if (active_joint_map_[j])
                {
                    joint_state_filtered.push_back(joint_states[i][j]);
                }
            }
            joint_states_filtered.push_back(joint_state_filtered);
        }

        return joint_states_filtered;
    }

    __global__ void getStepKernel(
        float * d_from_states,
        float * d_to_states,
        int num_of_config,
        int num_of_joints,
        float d_step_size,
        int * d_num_steps,
        float * move_direction,
        float * ditance_between_states
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_config) return;

        float * from_state = &d_from_states[idx * num_of_joints];
        float * to_state = &d_to_states[idx * num_of_joints];

        // calculate the distance between the two states
        float distance = 0.0f;
        for (size_t i = 0; i < num_of_joints; i++)
        {
            distance += (from_state[i] - to_state[i]) * (from_state[i] - to_state[i]);
        }
        distance = sqrt(distance);

        // set the distance between states
        ditance_between_states[idx] = distance;

        // calculate the number of steps
        int num_steps = (int)(distance / d_step_size) + 1;

        d_num_steps[idx] = num_steps;

        // calculate the move direction
        for (size_t i = 0; i < num_of_joints; i++)
        {
            move_direction[idx * num_of_joints + i] = (to_state[i] - from_state[i]) / num_steps;
        }
    }

    __global__ void calculateInterpolatedState(
        float * d_from_states,
        float * d_to_states,
        int num_of_config,
        int num_of_joints,
        int * d_motion_start_index,
        int * d_num_steps,
        float * d_move_direction,
        float * d_interpolated_states
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_config) return;

        // get the number of steps
        int num_steps = d_num_steps[idx];

        // get the start index
        int start_index = d_motion_start_index[idx];

        // get the from state
        float * from_state = &d_from_states[idx * num_of_joints];
        float * to_state = &d_to_states[idx * num_of_joints];

        // calculate the interpolated states
        for (size_t i = 0; i < num_steps; i++)
        {
            for (size_t j = 0; j < num_of_joints; j++)
            {
                d_interpolated_states[(start_index + i) * num_of_joints + j] = from_state[j] + d_move_direction[idx * num_of_joints + j] * i;
            }
        }

        // set the last state to the to state
        for (size_t j = 0; j < num_of_joints; j++)
        {
            d_interpolated_states[(start_index + num_steps - 1) * num_of_joints + j] = to_state[j];
        }
    }
    
    __global__ void getFeasibilityByTotalCostKernel(
        float * d_total_costs,
        int * d_feasibility,
        int num_of_motions,
        int * num_of_steps,
        int * motion_start_index
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_motions) return;

        // get the number of steps
        int num_steps = num_of_steps[idx];
        // get the start index
        int start_index = motion_start_index[idx];

        float total_cost = 0.0f;
        for (size_t i = 0; i < num_steps; i++)
        {
            total_cost += d_total_costs[start_index + i];
        }

        d_feasibility[idx] = (total_cost == 0.0f);
    }

    bool SingleArmSpace::checkMotions(
        const BaseStatesPtr & states1, 
        const BaseStatesPtr & states2, 
        std::vector<bool> & motion_feasibility,
        std::vector<float> & motion_costs
    )
    {
        size_t num_of_states1 = states1->getNumOfStates();
        size_t num_of_states2 = states2->getNumOfStates();
        if (num_of_states1 != num_of_states2)
        {
            // throw an exception
            throw std::runtime_error("Number of states in states1 and states2 are not equal");
        }
        if (num_of_states1 == 0)
        {
            // throw an exception
            throw std::runtime_error("No states to check");
        }

        motion_feasibility.resize(num_of_states1);
        motion_costs.resize(num_of_states1);

        // get space info
        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        // cast to SingleArmStatesPtr
        SingleArmStatesPtr single_arm_states1 = std::dynamic_pointer_cast<SingleArmStates>(states1);
        SingleArmStatesPtr single_arm_states2 = std::dynamic_pointer_cast<SingleArmStates>(states2);

        // get the joint states from the states
        float * d_joint_states1 = single_arm_states1->getJointStatesCuda();
        float * d_joint_states2 = single_arm_states2->getJointStatesCuda();
        int num_of_joints = space_info->num_of_joints;
        int * d_num_steps;
        int * d_motion_start_index;
        float * d_move_direction;
        float * d_distance_between_states;
        cudaMalloc(&d_num_steps, num_of_states1 * sizeof(int));
        cudaMalloc(&d_motion_start_index, num_of_states1 * sizeof(int));
        cudaMalloc(&d_move_direction, num_of_states1 * num_of_joints * sizeof(float));
        cudaMalloc(&d_distance_between_states, num_of_states1 * sizeof(float));

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states1 + threadsPerBlock - 1) / threadsPerBlock;

        // Calculate the number of steps and the move direction.
        getStepKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_joint_states1, 
            d_joint_states2, 
            num_of_states1, 
            num_of_joints,
            resolution_,
            d_num_steps,
            d_move_direction,
            d_distance_between_states
        );

        // wait for the kernel to finish with cuda check
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_num_steps(num_of_states1);
        cudaMemcpy(h_num_steps.data(), d_num_steps, num_of_states1 * sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<int> h_motion_start_index(num_of_states1);
        int total_num_steps = 0;
        for (size_t i = 0; i < h_num_steps.size(); i++)
        {
            h_motion_start_index[i] = total_num_steps;
            total_num_steps += h_num_steps[i];
        }

        // copy data to device memory
        cudaMemcpy(d_motion_start_index, h_motion_start_index.data(), num_of_states1 * sizeof(int), cudaMemcpyHostToDevice);

        // create interpolated states
        SingleArmStatesPtr interpolated_states = std::make_shared<SingleArmStates>(total_num_steps, space_info);

        if(! interpolated_states->isValid())
        {
            // print in red
            std::cerr << "\033[31m" << "Failed to allocate memory for interpolated states. " << "\033[0m" << std::endl;

            // set motion_feasibility to false
            motion_feasibility.assign(num_of_states1, false);
            // set motion_costs to 0
            motion_costs.assign(num_of_states1, 0.0f);

            // deallocate
            cudaFree(d_num_steps);
            cudaFree(d_motion_start_index);
            cudaFree(d_move_direction);
            cudaFree(d_distance_between_states);
            // deallocate interpolated_states
            interpolated_states.reset();

            return false;
        }

        float * d_interpolated_states = interpolated_states->getJointStatesCuda();

        // Calculate the interpolated states
        calculateInterpolatedState<<<blocksPerGrid, threadsPerBlock>>>(
            d_joint_states1, 
            d_joint_states2, 
            num_of_states1, 
            num_of_joints,
            d_motion_start_index,
            d_num_steps,
            d_move_direction,
            d_interpolated_states
        );

        // wait for the kernel to finish with cuda check
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // check the interpolated_states
        interpolated_states->update();
        checkStates(interpolated_states);

        // get the total costs from the interpolated states
        float * d_total_costs = interpolated_states->getTotalCostsCuda();
        int * d_motion_feasibility;
        cudaMalloc(&d_motion_feasibility, num_of_states1 * sizeof(int));

        // call kernel
        getFeasibilityByTotalCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_total_costs, 
            d_motion_feasibility, 
            num_of_states1,
            d_num_steps,
            d_motion_start_index
        );

        // wait for the kernel to finish with cuda check
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // copy data to host memory
        std::vector<int> h_motion_feasibility(num_of_states1);
        cudaMemcpy(h_motion_feasibility.data(), d_motion_feasibility, num_of_states1 * sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<float> h_distance_between_states(num_of_states1);
        cudaMemcpy(h_distance_between_states.data(), d_distance_between_states, num_of_states1 * sizeof(float), cudaMemcpyDeviceToHost);

        // calculate the sqrt difference between the two states
        for (size_t i = 0; i < num_of_states1; i++)
        {
            motion_feasibility[i] = h_motion_feasibility[i] == 0 ? false : true;
            motion_costs[i] = h_distance_between_states[i];
        }

        // deallocate
        cudaFree(d_num_steps);
        cudaFree(d_motion_start_index);
        cudaFree(d_move_direction);
        cudaFree(d_distance_between_states);
        cudaFree(d_motion_feasibility);

        // deallocate interpolated_states
        interpolated_states.reset();

        return true;
    }

    bool SingleArmSpace::oldCheckMotions(
        const BaseStatesPtr & states1, 
        const BaseStatesPtr & states2, 
        std::vector<bool> & motion_feasibility,
        std::vector<float> & motion_costs
    )
    {
        int num_of_states1 = states1->getNumOfStates();
        int num_of_states2 = states2->getNumOfStates();
        if (num_of_states1 != num_of_states2)
        {
            // throw an exception
            throw std::runtime_error("Number of states in states1 and states2 are not equal");
        }
        if (num_of_states1 == 0)
        {
            // throw an exception
            throw std::runtime_error("No states to check");
        }

        // static cast to SingleArmStatesPtr
        SingleArmStatesPtr single_arm_states1 = std::dynamic_pointer_cast<SingleArmStates>(states1);
        SingleArmStatesPtr single_arm_states2 = std::dynamic_pointer_cast<SingleArmStates>(states2);

        motion_feasibility.resize(num_of_states1);
        motion_costs.resize(num_of_states1);

        // get space info
        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        // get the joint states from the states
        std::vector<std::vector<float>> joint_states1 = single_arm_states1->getJointStatesFullHost();
        std::vector<std::vector<float>> joint_states2 = single_arm_states2->getJointStatesFullHost();

        std::vector<int> motion_start;
        std::vector<int> motion_end;
        std::vector<std::vector<float>> all_motions;

        for (int i = 0; i < num_of_states1; i++)
        {
            // get the interpolated states
            std::vector<std::vector<float>> interpolated_states = interpolateVectors(joint_states1[i], joint_states2[i], resolution_); 

            // calculate the sqrt difference between the two states
            float cost = 0.0f;
            for (size_t j = 0; j < joint_states1[i].size(); j++)
            {
                cost += (joint_states1[i][j] - joint_states2[i][j]) * (joint_states1[i][j] - joint_states2[i][j]);
            }
            motion_costs[i] = sqrt(cost);

            // motion_sizes.push_back(interpolated_states.size());
            motion_start.push_back(all_motions.size());
            motion_end.push_back(all_motions.size() + interpolated_states.size()); // exclusive
            all_motions.insert(all_motions.end(), interpolated_states.begin(), interpolated_states.end());
        }

        // create states from the all_motions
        auto interpolated_states = createStatesFromVectorFull(all_motions);
        if (interpolated_states == nullptr)
        {
            // print in red
            std::cerr << "\033[31m" << "Failed to allocate memory for interpolated states. " << "\033[0m" << std::endl;

            // set motion_feasibility to false
            motion_feasibility.assign(num_of_states1, false);
            // set motion_costs to 0
            motion_costs.assign(num_of_states1, 0.0f);

            return false;
        }

        interpolated_states->update();
        std::vector<bool> motion_state_feasibility;
        // check the interpolated_states
        checkStates(interpolated_states, motion_state_feasibility);

        // check the motion feasibility.
        for (int i = 0; i < num_of_states1; i++)
        {
            bool feasible = true;
            for (int j = motion_start[i]; j < motion_end[i]; j++)
            {
                if (!motion_state_feasibility[j])
                {
                    feasible = false;
                    break;
                }
            }
            motion_feasibility[i] = feasible;
        }

        // deallocate interpolated_states
        interpolated_states.reset();

        return true;
    }

    bool SingleArmSpace::checkConstrainedMotions(
        const BaseStatesPtr & states1, 
        const BaseStatesPtr & states2
    )
    {
        size_t num_of_states1 = states1->getNumOfStates();
        size_t num_of_states2 = states2->getNumOfStates();
        if (num_of_states1 != num_of_states2)
        {
            // throw an exception
            throw std::runtime_error("Number of states in states1 and states2 are not equal");
        }
        if (num_of_states1 == 0)
        {
            // throw an exception
            throw std::runtime_error("No states to check");
        }

        // get space info
        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        // cast to SingleArmStatesPtr
        SingleArmStatesPtr single_arm_states1 = std::dynamic_pointer_cast<SingleArmStates>(states1);
        SingleArmStatesPtr single_arm_states2 = std::dynamic_pointer_cast<SingleArmStates>(states2);

        /**
        Psuedo code:
            create a device memory for int to tell the current step moving to the next state. If it is -1, then it means this motion is impossible.
            int [] motion_step = 0 * [num_of_states1]
            float [] distance_to_states2 = floatMax * [num_of_states1]

            copy states1 to a intermediate_states

            call approachKernel on intermediate_states[i] to get closer to states2[i] if motion_step[i] != -1

            for loop: 0 to 5
                call forwardKinematicsKernel on intermediate_states[i] to get the forward kinematics if motion_step[i] != -1

                call computeGradAndErrorKernel on intermediate_states[i] to get the gradient and error if motion_step[i] != -1

                call update_with_grad on intermediate_states[i] to update the intermediate_states if motion_step[i] != -1

            check 




        
         */
    }

    void SingleArmSpace::interpolate(
        const BaseStatesPtr & from_states,
        const BaseStatesPtr & to_states,
        float max_distance
    )
    {
        // pass max_distance to device
        float * d_max_distance;
        cudaMalloc(&d_max_distance, sizeof(float));
        cudaMemcpy(d_max_distance, &max_distance, sizeof(float), cudaMemcpyHostToDevice);

        // static cast to SingleArmStatesPtr
        SingleArmStatesPtr single_arm_from_states = std::dynamic_pointer_cast<SingleArmStates>(from_states);
        SingleArmStatesPtr single_arm_to_states = std::dynamic_pointer_cast<SingleArmStates>(to_states);
        // get the joint states from the states in device
        float * d_from_states = single_arm_from_states->getJointStatesCuda();
        float * d_to_states = single_arm_to_states->getJointStatesCuda();

        int num_of_states = from_states->getNumOfStates();
        // call kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states + threadsPerBlock - 1) / threadsPerBlock;
        interpolate_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_from_states, 
            d_to_states, 
            num_of_states, 
            num_of_joints,
            d_max_distance
        );

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        // deallocate d_max_distance
        cudaFree(d_max_distance);
    }

    BaseStatesPtr SingleArmSpace::getPathFromWaypoints(
        const BaseStatesPtr & waypoints
    )
    {
        int num_of_waypoints = waypoints->getNumOfStates();

        // static cast to SingleArmStatesPtr
        SingleArmStatesPtr single_arm_waypoints = std::dynamic_pointer_cast<SingleArmStates>(waypoints);

        // get space info
        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        // get the joint states from the states
        std::vector<std::vector<float>> waypoints_joint_values = single_arm_waypoints->getJointStatesFullHost();

        std::vector<std::vector<float>> path_in_host;
        for (int i = 0; i < num_of_waypoints - 1; i++)
        {
            // get the interpolated states
            std::vector<std::vector<float>> interpolated_states = interpolateVectors(waypoints_joint_values[i], waypoints_joint_values[i+1], resolution_); 
            path_in_host.insert(path_in_host.end(), interpolated_states.begin(), interpolated_states.end());
        }

        // create states from the all_motions
        auto path_in_cuda = createStatesFromVectorFull(path_in_host);
        if( path_in_cuda == nullptr)
        {
            // print in red
            std::cerr << "\033[31m" << "Failed to allocate memory for path in cuda. " << "\033[0m" << std::endl;

            // return empty pointer
            return nullptr;
        }

        return path_in_cuda;
    }

    void SingleArmSpace::oldCheckStates(
        const BaseStatesPtr & states,
        std::vector<bool>& state_feasibility
    )
    {
        this->oldCheckStates(states);

        std::vector<float> total_costs = states->getTotalCostsHost();

        state_feasibility.assign(total_costs.size(), false);

        for (size_t i = 0; i < total_costs.size(); i++)
        {
            state_feasibility[i] = (total_costs[i] == 0.0f);
        }
    }

    void SingleArmSpace::checkStates(
        const BaseStatesPtr & states,
        std::vector<bool>& state_feasibility
    )
    {
        this->checkStates(states);

        std::vector<float> total_costs = states->getTotalCostsHost();

        state_feasibility.assign(total_costs.size(), false);

        for (size_t i = 0; i < total_costs.size(); i++)
        {
            state_feasibility[i] = (total_costs[i] == 0.0f);
        }
    }

    void SingleArmSpace::oldCheckStates(const BaseStatesPtr & states)
    {
        // based on all the constraints, check if the states are feasible
        for (size_t i = 0; i < constraints_.size(); i++)
        {
            // auto start_time = std::chrono::high_resolution_clock::now();
            constraints_[i]->computeCostLarge(states);
            // auto end_time = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> elapsed_seconds = end_time - start_time;
            // std::cout << "Constraint " << constraints_[i]->getName() << " took: " << elapsed_seconds.count() << "s" << std::endl;
        }

        // get the total cost
        states->calculateTotalCosts();
    }

    void SingleArmSpace::checkStates(const BaseStatesPtr & states)
    {
        // based on all the constraints, check if the states are feasible
        for (size_t i = 0; i < constraints_.size(); i++)
        {
            // constraints_[i]->computeCost(states);
            constraints_[i]->computeCostFast(states);
        }

        // get the total cost
        states->calculateTotalCosts();
    }

    __global__ void update_with_grad(
        float * d_states,
        float * d_grad,
        float step_size,
        int num_of_states,
        int num_of_joints,
        int * d_active_joint_map
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_states * num_of_joints) return;

        int joint_idx = idx % num_of_joints;

        // if joint is not active, then set the value to 0
        if (d_active_joint_map[joint_idx] == 0)
        {
            d_states[idx] = d_states[idx];
        }
        else
        {
            d_states[idx] = d_states[idx] + step_size * d_grad[idx];
        }
    }

    void SingleArmSpace::projectStates(BaseStatesPtr states)
    {
        // cast to SingleArmStatesPtr
        SingleArmStatesPtr single_arm_states = std::dynamic_pointer_cast<SingleArmStates>(states);

        for (int t = 0; t < CUDAMPLib_PROJECT_MAX_ITERATION; t++)
        {
            // forward kinematics
            single_arm_states->calculateForwardKinematics();

            for (int i : projectable_constraint_indices_)
            {
                if (constraints_[i]->isProjectable())
                {
                    constraints_[i]->computeGradientAndError(single_arm_states);
                }
                else{
                    // raise an exception
                    throw std::runtime_error("Constraint " + constraints_[i]->getName() + " is not projectable");
                }
            }

            single_arm_states->calculateTotalGradientAndError(projectable_constraint_indices_);

            // print the total gradient and error
            // float * d_total_costs = single_arm_states->getTotalCostsCuda(); // [num_of_states]

            // float * d_total_gradient = single_arm_states->getTotalGradientCuda(); // [num_of_states * num_of_joints]

            // std::cout << "Iteration: " << t << std::endl;

            // // print joint values
            // std::vector<std::vector<float>> joint_values = single_arm_states->getJointStatesFullHost();
            // std::cout << "Joint values" << std::endl;
            // for (size_t i = 0; i < joint_values.size(); i++)
            // {
            //     for (size_t j = 0; j < joint_values[i].size(); j++)
            //     {
            //         std::cout << joint_values[i][j] << " ";
            //     }
            //     std::cout << std::endl;
            // }

            // // print total gradient
            // std::vector<float> total_gradient(single_arm_states->getNumOfStates() * single_arm_states->getNumOfJoints(), 0.0);
            // cudaMemcpy(total_gradient.data(), d_total_gradient, single_arm_states->getNumOfStates() * single_arm_states->getNumOfJoints() * sizeof(float), cudaMemcpyDeviceToHost);
            // std::cout << "Total gradient" << std::endl;
            // for (size_t i = 0; i < total_gradient.size(); i++)
            // {
            //     std::cout << total_gradient[i] << " ";
            //     if ((i + 1) % single_arm_states->getNumOfJoints() == 0)
            //     {
            //         std::cout << std::endl;
            //     }
            // }

            // // print total costs
            // std::vector<float> total_costs(single_arm_states->getNumOfStates(), 0.0);
            // cudaMemcpy(total_costs.data(), d_total_costs, single_arm_states->getNumOfStates() * sizeof(float), cudaMemcpyDeviceToHost);
            // std::cout << "Total costs" << std::endl;
            // for (size_t i = 0; i < total_costs.size(); i++)
            // {
            //     std::cout << total_costs[i] << " ";
            // }
            // std::cout << std::endl;

            // update the states
            int threadsPerBlock = 256;
            int blocksPerGrid = (single_arm_states->getNumOfStates() * single_arm_states->getNumOfJoints() + threadsPerBlock - 1) / threadsPerBlock;
            update_with_grad<<<blocksPerGrid, threadsPerBlock>>>(
                single_arm_states->getJointStatesCuda(),
                single_arm_states->getTotalGradientCuda(),
                1.0,
                single_arm_states->getNumOfStates(),
                single_arm_states->getNumOfJoints(),
                d_active_joint_map
            );

            // wait for the kernel to finish
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    void SingleArmSpace::getSpaceInfo(SingleArmSpaceInfoPtr space_info)
    {
        // call the base class function
        BaseSpace::getSpaceInfo(space_info);

        // set the additional information for single arm space
        space_info->d_joint_types = d_joint_types;
        space_info->d_joint_poses = d_joint_poses;
        space_info->d_joint_axes = d_joint_axes;
        space_info->d_link_parent_link_maps = d_link_parent_link_maps;
        space_info->d_collision_spheres_to_link_map = d_collision_spheres_to_link_map;
        space_info->d_self_collision_spheres_pos_in_link = d_self_collision_spheres_pos_in_link;
        space_info->d_self_collision_spheres_radius = d_self_collision_spheres_radius;
        space_info->d_active_joint_map = d_active_joint_map;
        space_info->d_lower_bound = d_lower_bound;
        space_info->d_upper_bound = d_upper_bound;
        space_info->d_default_joint_values = d_default_joint_values;

        space_info->num_of_joints = num_of_joints;
        space_info->num_of_links = num_of_links;
        space_info->num_of_self_collision_spheres = num_of_self_collision_spheres;
        space_info->link_names = link_names_;

        // set the bounds
        space_info->lower_bound = lower_bound;
        space_info->upper_bound = upper_bound;

        // set number of active joints
        space_info->num_of_active_joints = num_of_active_joints_;
        space_info->active_joint_map = active_joint_map_;
    }

    BaseStateManagerPtr SingleArmSpace::createStateManager()
    {
        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        return std::make_shared<SingleArmStateManager>(space_info);
    }
} // namespace cudampl
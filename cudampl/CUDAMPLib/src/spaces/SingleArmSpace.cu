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
        // check if space contains projectable constraints
        if (projectable_constraint_indices_.size() > 0)
        {
            return checkConstrainedMotions(states1, states2, motion_feasibility, motion_costs);
        }
        else
        {
            return checkUnconstrainedMotions(states1, states2, motion_feasibility, motion_costs);
        }
    }

    bool SingleArmSpace::checkUnconstrainedMotions(
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

    __global__ void approachKernel(
        int * d_is_approaching,
        float * d_joint_states1,
        float * d_joint_states2,
        float * d_approach_directions,
        int num_of_states,
        int num_of_joints,
        float approach_step_size
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_states) return;

        if (d_is_approaching[idx] == 0) return;

        float * joint_states1 = &d_joint_states1[idx * num_of_joints];
        float * joint_states2 = &d_joint_states2[idx * num_of_joints];
        float * approach_directions = &d_approach_directions[idx * num_of_joints];

        // Compute differences and accumulate squared norm in one loop.
        float norm = 0.0f;
        for (int i = 0; i < num_of_joints; i++) {
            float diff = joint_states2[i] - joint_states1[i];
            approach_directions[i] = diff;
            norm += diff * diff;
        }

        norm = sqrtf(norm);

        // Only normalize and update if the norm is significant.
        if (norm > 1e-6f) {
            for (int i = 0; i < num_of_joints; i++) {
                float normalized = approach_directions[i] / norm;
                approach_directions[i] = normalized;
                joint_states1[i] += normalized * approach_step_size;
            }
        }
    }

    /**
        Check if the approaching is valid. 
        It is valid is only
        1. the new intermediate state is closer to the goal state.
        2. the new intermediate state is not too far from previous intermediate state. It is less than max_step_size.
     */
    __global__ void checkValidOfApproaching(
        float * d_previous_intermediate_joint_states,
        float * d_current_intermediate_joint_states,
        float * d_goal_joint_states2,
        int num_of_states,
        int num_of_joints,
        float max_step_size,
        int * result
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_states) return;

        if (result[idx] == 0) return; // if the result is already 0, then return

        float * previous_intermediate_joint_states = &d_previous_intermediate_joint_states[idx * num_of_joints];
        float * current_intermediate_joint_states = &d_current_intermediate_joint_states[idx * num_of_joints];
        float * goal_joint_states2 = &d_goal_joint_states2[idx * num_of_joints];

        float distance_from_previous_to_goal = 0.0f;
        float distance_from_current_to_goal = 0.0f;
        float distance_from_previous_to_current = 0.0f;

        // calculate the distance from previous to goal
        for (size_t i = 0; i < num_of_joints; i++)
        {
            distance_from_previous_to_goal += (previous_intermediate_joint_states[i] - goal_joint_states2[i]) * (previous_intermediate_joint_states[i] - goal_joint_states2[i]);
        }
        distance_from_previous_to_goal = sqrt(distance_from_previous_to_goal);

        // calculate the distance from current to goal
        for (size_t i = 0; i < num_of_joints; i++)
        {
            distance_from_current_to_goal += (current_intermediate_joint_states[i] - goal_joint_states2[i]) * (current_intermediate_joint_states[i] - goal_joint_states2[i]);
        }
        distance_from_current_to_goal = sqrt(distance_from_current_to_goal);

        // calculate the distance from previous to current
        for (size_t i = 0; i < num_of_joints; i++)
        {
            distance_from_previous_to_current += (previous_intermediate_joint_states[i] - current_intermediate_joint_states[i]) * (previous_intermediate_joint_states[i] - current_intermediate_joint_states[i]);
        }
        distance_from_previous_to_current = sqrt(distance_from_previous_to_current);

        if (distance_from_current_to_goal > distance_from_previous_to_goal)
        {
            result[idx] = 0;
        }

        if (distance_from_previous_to_current > max_step_size)
        {
            result[idx] = 0;
        }
    }

    __global__ void update_with_grad_and_flag(
        int * d_is_approaching,
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

        if (d_is_approaching[idx / num_of_joints] == 0) return;

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

    /**
        If the state is approaching the goal state, then update the motion step.
     */
    __global__ void updateMotionStep(
        int * d_is_approaching,
        int * d_motion_step,
        int num_of_states
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_of_states) return;

        if (d_is_approaching[idx] == 0) return; // if the state is not approaching, then return

        int motion_step = d_motion_step[idx];

        d_motion_step[idx] = motion_step + 1;
    }

    /**
        Check if the state has achieved the goal state.
     */
    __global__ void checkAchieveGoal(
        int * d_is_approaching,
        float * d_joint_intermediate_states,
        float * d_joint_goal_states,
        int num_of_states,
        int num_of_joints,
        float distance_threshold,
        int * d_achieve_goal
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_states) return;

        if (d_is_approaching[idx] == 0) return; // if the state is not approaching, then return

        float distance = 0.0f;
        for (size_t i = 0; i < num_of_joints; i++)
        {
            distance += (d_joint_intermediate_states[idx * num_of_joints + i] - d_joint_goal_states[idx * num_of_joints + i]) * (d_joint_intermediate_states[idx * num_of_joints + i] - d_joint_goal_states[idx * num_of_joints + i]);
        }
        distance = sqrt(distance);

        if (distance < distance_threshold)
        {
            // set the achieve flag to 1
            d_achieve_goal[idx] = 1;
            // set the is_approaching to 0
            d_is_approaching[idx] = 0;
        }
    }

    __global__ void checkIfCostIsZero(
        float * d_total_costs,
        int num_of_states,
        float cost_threshold,
        int * result
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_states) return;
        if (result[idx] == 0) return;

        if (d_total_costs[idx] > cost_threshold)
        {
            result[idx] = 0;
        }
    }

    std::vector<std::vector<std::vector<float>>> SingleArmSpace::computeConstrainedMotions(
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

        motion_feasibility.resize(num_of_states1, false);
        motion_costs.resize(num_of_states1, 0.0f);

        // get space info
        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        // cast to SingleArmStatesPtr
        SingleArmStatesPtr single_arm_states1 = std::dynamic_pointer_cast<SingleArmStates>(states1);
        SingleArmStatesPtr single_arm_states2 = std::dynamic_pointer_cast<SingleArmStates>(states2);

        int * d_motion_step; // Store the number of steps the state has moved
        cudaMalloc(&d_motion_step, num_of_states1 * sizeof(int));
        cudaMemset(d_motion_step, 0, num_of_states1 * sizeof(int));

        int * d_is_approaching; // the flag indicates if the state is approaching the goal state
        cudaMalloc(&d_is_approaching, num_of_states1 * sizeof(int));
        cudaMemset(d_is_approaching, 1, num_of_states1 * sizeof(int));

        float * d_distance_to_states2; // Store the distance to the goal states
        cudaMalloc(&d_distance_to_states2, num_of_states1 * sizeof(float));
        float * d_new_distance_to_states2; // Store the new distance to the goal states
        cudaMalloc(&d_new_distance_to_states2, num_of_states1 * sizeof(float));

        float * d_approach_directions; // Store the approach directions
        cudaMalloc(&d_approach_directions, num_of_states1 * num_of_joints * sizeof(float));

        int * d_achieve_goal; // Store the flag if the state has achieved the goal
        cudaMalloc(&d_achieve_goal, num_of_states1 * sizeof(int));
        cudaMemset(d_achieve_goal, 0, num_of_states1 * sizeof(int));

        float * d_joint_states1 = single_arm_states1->getJointStatesCuda();
        float * d_joint_states2 = single_arm_states2->getJointStatesCuda();

        // Create intermediate_states
        SingleArmStatesPtr intermediate_states = std::make_shared<SingleArmStates>(num_of_states1, space_info);
        float * d_joint_intermediate_states = intermediate_states->getJointStatesCuda();

        // Create device memory to hold the previous intermediate states
        float * d_joint_previous_intermediate_states;
        cudaMalloc(&d_joint_previous_intermediate_states, num_of_states1 * num_of_joints * sizeof(float));

        // copy the joint_states1 to d_joint_intermediate_states
        cudaMemcpy(d_joint_intermediate_states, d_joint_states1, num_of_states1 * num_of_joints * sizeof(float), cudaMemcpyDeviceToDevice);

        // constrained motions
        std::vector<std::vector<std::vector<float>>> constrained_motions(num_of_states1, std::vector<std::vector<float>>());
        std::vector<float> approaching_joint_states_flatten(num_of_states1 * num_of_joints, 0.0);
        std::vector<int> h_is_approaching(num_of_states1);
        std::vector<std::vector<float>> intermediate_states_joint_values(num_of_states1, std::vector<float>(num_of_joints, 0.0));

        // calculate the distance to states2
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states1 + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGrid2 = (num_of_states1 * num_of_joints + threadsPerBlock - 1) / threadsPerBlock;

        for (size_t ii = 0; ii < 10000; ii++)
        {
            // store the previous intermediate states
            cudaMemcpy(d_joint_previous_intermediate_states, d_joint_intermediate_states, num_of_states1 * num_of_joints * sizeof(float), cudaMemcpyDeviceToDevice);

            // call approachKernel on intermediate_states[i] to get closer to states2[i] if is_approaching[i] == 1
            approachKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_is_approaching, // If is_approaching[i] == 0, then do not update the intermediate_states[i]
                d_joint_intermediate_states,
                d_joint_states2,
                d_approach_directions,
                num_of_states1,
                num_of_joints,
                resolution_
            );

            // project the intermediate_states
            for (size_t t = 0; t < 2; t++)
            {
                // forward kinematics
                intermediate_states->calculateForwardKinematics();

                for (int i : projectable_constraint_indices_)
                {
                    if (constraints_[i]->isProjectable())
                    {
                        constraints_[i]->computeGradientAndError(intermediate_states);
                    }
                    else{
                        // raise an exception
                        throw std::runtime_error("Constraint " + constraints_[i]->getName() + " is not projectable");
                    }
                }

                // calculate the total gradient
                intermediate_states->calculateTotalGradientAndError(projectable_constraint_indices_);

                // update with gradient
                update_with_grad_and_flag<<<blocksPerGrid2, threadsPerBlock>>>(
                    d_is_approaching,
                    d_joint_intermediate_states,
                    intermediate_states->getTotalGradientCuda(),
                    1.0,
                    num_of_states1,
                    num_of_joints,
                    d_active_joint_map
                );

                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // check if the projected states still satisfy the constraints
            checkIfCostIsZero<<<blocksPerGrid, threadsPerBlock>>>(
                intermediate_states->getTotalCostsCuda(),
                num_of_states1,
                1e-3,
                d_is_approaching
            );

            // check if the projected states is closer to states2
            checkValidOfApproaching<<<blocksPerGrid, threadsPerBlock>>>(
                d_joint_previous_intermediate_states,
                d_joint_intermediate_states,
                d_joint_states2,
                num_of_states1,
                num_of_joints,
                resolution_ * 5.0,
                d_is_approaching
            );

            CUDA_CHECK(cudaDeviceSynchronize());

            // increase the motion step if the states are still approaching
            updateMotionStep<<<blocksPerGrid, threadsPerBlock>>>(d_is_approaching, d_motion_step, num_of_states1);

            CUDA_CHECK(cudaDeviceSynchronize());

            // check if ths new intermediate states is close enough to the goal states.
            checkAchieveGoal<<<blocksPerGrid, threadsPerBlock>>>(
                d_is_approaching,
                d_joint_intermediate_states,
                d_joint_states2,
                num_of_states1,
                num_of_joints,
                resolution_,
                d_achieve_goal
            );

            CUDA_CHECK(cudaDeviceSynchronize());

            // copy data to host memory  
            cudaMemcpyAsync(h_is_approaching.data(), d_is_approaching, num_of_states1 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpyAsync(
                approaching_joint_states_flatten.data(), 
                d_joint_intermediate_states, 
                num_of_states1 * num_of_joints * sizeof(float), 
                cudaMemcpyDeviceToHost
            );
            
            bool no_states_to_approach = true;

            CUDA_CHECK(cudaDeviceSynchronize());

            for (size_t i = 0; i < h_is_approaching.size(); i++)
            {
                if (h_is_approaching[i] != 0)
                {
                    no_states_to_approach = false;
                    break;
                }
            }

            // Reshape the joint states
            for (size_t i = 0; i < num_of_states1; i++)
            {
                // std::cout << "intermediate_states[" << i << "]: ";
                std::vector<float> current_joint_values(num_of_joints, 0.0);
                for (int j = 0; j < num_of_joints; j++)
                {
                    current_joint_values[j] = approaching_joint_states_flatten[i * num_of_joints + j];
                    // std::cout << current_joint_values[j] << " ";
                }
                // std::cout << std::endl;

                // append to the constrained_motions
                constrained_motions[i].push_back(current_joint_values);
            }

            if (no_states_to_approach)
            {
                break;
            }
        }

        std::vector<int> h_achieve_goal_int(num_of_states1);
        cudaMemcpy(h_achieve_goal_int.data(), d_achieve_goal, num_of_states1 * sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<int> h_motion_step(num_of_states1);
        cudaMemcpy(h_motion_step.data(), d_motion_step, num_of_states1 * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_motion_step);
        cudaFree(d_is_approaching);
        cudaFree(d_new_distance_to_states2);
        cudaFree(d_approach_directions);
        cudaFree(d_joint_previous_intermediate_states);
        cudaFree(d_achieve_goal);
        // deallocate intermediate_states
        intermediate_states.reset();

        ////////////////////////////////////////// Check the feasibility of the constrained motions //////////////////////////////////////////

        // allocate memory based on the total number of step from h_motion_step
        int total_steps = 0;
        std::vector<int> motion_start_index(num_of_states1);
        for (size_t i = 0; i < h_motion_step.size(); i++)
        {
            motion_start_index[i] = total_steps;
            total_steps += h_motion_step[i];
        }

        if (total_steps == 0)
        {
            // set motion_feasibility to false
            motion_feasibility.assign(num_of_states1, false);
            
            // return empty std::vector<std::vector<std::vector<float>>>
            return std::vector<std::vector<std::vector<float>>>();
        }

        // allocate memory for the interpolated states
        SingleArmStatesPtr interpolated_states = std::make_shared<SingleArmStates>(total_steps, space_info);
        float * d_joint_values_interpolated_states = interpolated_states->getJointStatesCuda();

        for (size_t i = 0; i < num_of_states1; i++)
        {
            if (h_achieve_goal_int[i] != 0){
                // for motion i
                for (int j = 0; j < h_motion_step[i]; j++)
                {
                    // get the joint values
                    std::vector<float> joint_values = constrained_motions[i][j];
                    // copy the joint values to the d_joint_values_interpolated_states
                    cudaMemcpyAsync(d_joint_values_interpolated_states + (motion_start_index[i] + j) * num_of_joints, joint_values.data(), num_of_joints * sizeof(float), cudaMemcpyHostToDevice);
                }
            }
        }
        // wait copy to finish
        CUDA_CHECK(cudaDeviceSynchronize());

        // check the interpolated_states
        interpolated_states->update();
        std::vector<bool> motion_state_feasibility;
        checkStates(interpolated_states, motion_state_feasibility);

        std::vector<std::vector<std::vector<float>>> cropped_constrained_motions(num_of_states1, std::vector<std::vector<float>>());

        for (size_t i = 0; i < num_of_states1; i++)
        {
            if (h_achieve_goal_int[i] != 0)
            {
                // get the motion start index
                int start_index = motion_start_index[i];
                bool is_feasible_motion = true;
                for (int j = 0; j < h_motion_step[i]; j++)
                {
                    // std::cout << motion_state_feasibility[start_index + j] << " ";
                    if (!motion_state_feasibility[start_index + j])
                    {
                        is_feasible_motion = false;
                    }
                }
                // std::cout << std::endl;
                motion_feasibility[i] = is_feasible_motion;

                if (is_feasible_motion)
                {
                    cropped_constrained_motions[i].push_back(constrained_motions[i][0]);

                    // print the feasible motion
                    float total_cost = 0.0;
                    for (int j = 1; j < h_motion_step[i]; j++)
                    {
                        float cost = 0.0;
                        for (int k = 0; k < num_of_joints; k++)
                        {
                            cost += (constrained_motions[i][j][k] - constrained_motions[i][j-1][k]) * (constrained_motions[i][j][k] - constrained_motions[i][j-1][k]);
                        }
                        total_cost += sqrt(cost);;

                        cropped_constrained_motions[i].push_back(constrained_motions[i][j]);
                    }

                    motion_costs[i] = total_cost;
                }
            }
        }

        // free the memory
        interpolated_states.reset();
    
        return cropped_constrained_motions;
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

        if (projectable_constraint_indices_.size() > 0)
        {
            // This space has projectable constraints, so we need to project the sampled states first.
            this->projectStates(to_states);
        }

        // deallocate d_max_distance
        cudaFree(d_max_distance);
    }

    BaseStatesPtr SingleArmSpace::getPathFromWaypoints(
        const BaseStatesPtr & waypoints
    )
    {
        // check if space contains projectable constraints
        if (projectable_constraint_indices_.size() > 0)
        {
            // This space has projectable constraints, so we need to project the sampled states first.
            return getConstrainedPathFromWaypoints(waypoints);
        }
        else{
            return getUnconstrainedPathFromWaypoints(waypoints);
        }


    }

    BaseStatesPtr SingleArmSpace::getUnconstrainedPathFromWaypoints(
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

    bool SingleArmSpace::checkConstrainedMotions(
        const BaseStatesPtr & states1, 
        const BaseStatesPtr & states2,
        std::vector<bool>& motion_feasibility,
        std::vector<float>& motion_costs
    )
    {
        // call computeConstrainedMotions. Ignore the return value
        computeConstrainedMotions(states1, states2, motion_feasibility, motion_costs);

        return true;
    }

    BaseStatesPtr SingleArmSpace::getConstrainedPathFromWaypoints(
        const BaseStatesPtr & waypoints
    )
    {
        // static cast to SingleArmStatesPtr
        SingleArmStatesPtr single_arm_waypoints = std::dynamic_pointer_cast<SingleArmStates>(waypoints);

        // Due to constrained motion is not necessary valid in both directions, we need to check the feasibility of the motion
        // in two directions. 

        size_t num_of_waypoints = single_arm_waypoints->getNumOfStates();

        std::vector<std::vector<float>> waypoints_joint_values = single_arm_waypoints->getJointStatesHost();

        std::vector<std::vector<float>> states1_joint_values;
        std::vector<std::vector<float>> states2_joint_values;

        for (size_t i = 0 ; i < num_of_waypoints - 1; i++)
        {
            // forward motion
            states1_joint_values.push_back(waypoints_joint_values[i]);
            states2_joint_values.push_back(waypoints_joint_values[i+1]);

            // backward motion
            states1_joint_values.push_back(waypoints_joint_values[i+1]);
            states2_joint_values.push_back(waypoints_joint_values[i]);
        }

        // create states from the states1_joint_values and states2_joint_values
        auto states1 = createStatesFromVector(states1_joint_values);
        auto states2 = createStatesFromVector(states2_joint_values);
        std::vector<bool> motion_feasibility;
        std::vector<float> motion_costs;

        // check the constrained motions
        std::vector<std::vector<std::vector<float>>> motion_joint_values = computeConstrainedMotions(states1, states2, motion_feasibility, motion_costs);

        // // print motion joint values
        // for (size_t i = 0; i < motion_joint_values.size(); i++)
        // {
        //     std::cout << "Motion " << i << " feasibility: " << motion_feasibility[i] << std::endl;
        //     for (size_t j = 0; j < motion_joint_values[i].size(); j++)
        //     {
        //         for (size_t k = 0; k < motion_joint_values[i][j].size(); k++)
        //         {
        //             std::cout << motion_joint_values[i][j][k] << " ";
        //         }
        //         std::cout << std::endl;
        //     }
        // }

        std::vector<std::vector<float>> path_joint_values;

        for (size_t i = 0; i < motion_joint_values.size(); i += 2)
        {
            if (motion_feasibility[i])
            {
                for (size_t j = 0; j < motion_joint_values[i].size(); j++)
                {
                    path_joint_values.push_back(motion_joint_values[i][j]);
                }
                continue;
            }
            if (motion_feasibility[i+1])
            {
                // add joint values in reverse order
                for (int j = motion_joint_values[i+1].size() - 1; j >= 0; j--)
                {
                    path_joint_values.push_back(motion_joint_values[i+1][j]);
                }
                continue;
            }
        }

        // // print path_joint_values
        // for (size_t i = 0; i < path_joint_values.size(); i++)
        // {
        //     for (size_t j = 0; j < path_joint_values[i].size(); j++)
        //     {
        //         std::cout << path_joint_values[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        auto path_states = createStatesFromVectorFull(path_joint_values);

        // deallocate states1 and states2
        states1.reset();
        states2.reset();
        
        return path_states;
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
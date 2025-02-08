#include <spaces/SingleArmSpace.h>

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
        const std::vector<float>& default_joint_values
    )
        : BaseSpace(dim, constraints),
          gen(std::random_device{}()),
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
            d_sampled_states[idx] = curand_uniform(&local_state);// * (d_upper_bound[joint_idx] - d_lower_bound[joint_idx]) + d_lower_bound[joint_idx];
        }
    }

    BaseStatesPtr SingleArmSpace::sample(int num_of_config)
    {
        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        // Create a state
        SingleArmStatesPtr sampled_states = std::make_shared<SingleArmStates>(num_of_config, space_info);

        // get device memory with size of num_of_config * num_of_joints * sizeof(float)
        float * d_sampled_states = sampled_states->getJointStatesCuda();

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_config * num_of_joints + threadsPerBlock - 1) / threadsPerBlock;

        // // set random seed
        // std::random_device rd;  // Non-deterministic seed (preferred)
        // std::mt19937_64 gen(rd()); // 64-bit Mersenne Twister PRNG
        // std::uniform_int_distribution<unsigned long> dist(0, ULONG_MAX);

        unsigned long seed = dist(gen);
        curandState * d_random_state;
        cudaMalloc(&d_random_state, num_of_config * num_of_joints * sizeof(curandState));
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

        return sampled_states;
    }

    BaseStatesPtr SingleArmSpace::createStatesFromVectorFull(const std::vector<std::vector<float>>& joint_values)
    {
        int num_of_config = joint_values.size();

        if (num_of_config == 0)
        {
            // throw an exception
            throw std::runtime_error("No joint values is empty");
        }

        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        // Create a state
        SingleArmStatesPtr generated_states = std::make_shared<SingleArmStates>(num_of_config, space_info);

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
                throw std::runtime_error("Joint values size is not correct");
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

        // get device memory with size of num_of_config * num_of_joints * sizeof(float)
        float * d_generated_states = generated_states->getJointStatesCuda();

        // copy data to device memory
        cudaMemcpy(d_generated_states, floatVectorFlatten(joint_value_w_correct_size).data(), num_of_config * num_of_joints * sizeof(float), cudaMemcpyHostToDevice);

        return generated_states;
    }

    void SingleArmSpace::checkMotions(
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
        std::vector<std::vector<float>> joint_states1 = single_arm_states1->getJointStatesHost();
        std::vector<std::vector<float>> joint_states2 = single_arm_states2->getJointStatesHost();

        std::vector<int> motion_start;
        std::vector<int> motion_end;
        std::vector<std::vector<float>> all_motions;

        for (int i = 0; i < num_of_states1; i++)
        {
            // get the interpolated states
            std::vector<std::vector<float>> interpolated_states = interpolateVectors(joint_states1[i], joint_states2[i], 10); 
            // TODO: the step size is too naive, need to be improved

            // calculate the sqrt difference between the two states
            float cost = 0.0f;
            for (int j = 0; j < joint_states1[i].size(); j++)
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
        interpolated_states->update();
        std::vector<bool> motion_state_feasibility;
        // check the interpolated_states
        checkStates(interpolated_states, motion_state_feasibility);

        // check the motion feasibility. TODO: This can be done in parallel
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
    }

    void SingleArmSpace::checkMotions(
        const BaseMotionsPtr & motions,
        std::vector<bool>& motion_feasibility,
        std::vector<float>& motion_costs
    ) 
    {
        if (motions->getNumOfMotions() == 0)
        {
            // throw an exception
            throw std::runtime_error("No motions to check");
        }

        motion_feasibility.clear();
        motion_costs.clear();

        // get space info
        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        SingleArmMotionsPtr single_arm_motions = std::dynamic_pointer_cast<SingleArmMotions>(motions);

        std::vector<std::vector<float>> motion_end_states1 = single_arm_motions->getJointStates1Host();
        std::vector<std::vector<float>> motion_end_states2 = single_arm_motions->getJointStates2Host();

        std::vector<int> motion_start;
        std::vector<int> motion_end;
        std::vector<std::vector<float>> all_motions;

        // print the motion end states
        for (int i = 0; i < motions->getNumOfMotions(); i++)
        {
            // get the interpolated states
            std::vector<std::vector<float>> interpolated_states = interpolateVectors(motion_end_states1[i], motion_end_states2[i], 10); 
            // TODO: the step size is too naive, need to be improved

            // calculate the sqrt difference between the two states
            float cost = 0.0f;
            for (int j = 0; j < motion_end_states1[i].size(); j++)
            {
                cost += (motion_end_states1[i][j] - motion_end_states2[i][j]) * (motion_end_states1[i][j] - motion_end_states2[i][j]);
            }
            motion_costs.push_back(sqrt(cost));

            // motion_sizes.push_back(interpolated_states.size());
            motion_start.push_back(all_motions.size());
            motion_end.push_back(all_motions.size() + interpolated_states.size()); // exclusive
            all_motions.insert(all_motions.end(), interpolated_states.begin(), interpolated_states.end());
        }

        // create states from the all_motions
        auto interpolated_states = createStatesFromVectorFull(all_motions);
        interpolated_states->update();
        std::vector<bool> motion_state_feasibility;
        // check the interpolated_states
        checkStates(interpolated_states, motion_state_feasibility);

        // deallocate interpolated_states
        interpolated_states.reset();

        // check the motion feasibility. TODO: This can be done in parallel
        for (int i = 0; i < motions->getNumOfMotions(); i++)
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
            motion_feasibility.push_back(feasible);
        }
    }

    void SingleArmSpace::checkStates(
        const BaseStatesPtr & states,
        std::vector<bool>& state_feasibility
    )
    {
        // based on all the constraints, check if the states are feasible
        for (size_t i = 0; i < constraints.size(); i++)
        {
            BaseConstraintPtr constraint = constraints[i];
            constraint->computeCost(states);
        }

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        // get the total cost
        states->calculateTotalCosts();

        std::vector<float> total_costs = states->getTotalCostsHost();

        for (size_t i = 0; i < total_costs.size(); i++)
        {
            state_feasibility.push_back(total_costs[i] == 0.0f);
        }
    }

    void SingleArmSpace::checkStates(const BaseStatesPtr & states)
    {
        // based on all the constraints, check if the states are feasible
        for (size_t i = 0; i < constraints.size(); i++)
        {
            BaseConstraintPtr constraint = constraints[i];
            constraint->computeCost(states);
        }

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        // get the total cost
        states->calculateTotalCosts();
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

        // set the bounds
        space_info->lower_bound = lower_bound;
        space_info->upper_bound = upper_bound;
    }

    BaseGraphPtr SingleArmSpace::createGraph()
    {
        return std::make_shared<SingleArmGraph>(num_of_joints);
    }

    BaseStateManagerPtr SingleArmSpace::createStateManager()
    {
        SingleArmSpaceInfoPtr space_info = std::make_shared<SingleArmSpaceInfo>();
        getSpaceInfo(space_info);

        return std::make_shared<SingleArmStateManager>(space_info);
    }
} // namespace cudampl
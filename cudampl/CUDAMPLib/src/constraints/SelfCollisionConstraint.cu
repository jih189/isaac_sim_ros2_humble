#include <constraints/SelfCollisionConstraint.h>

#include <chrono>

namespace CUDAMPLib{

    SelfCollisionConstraint::SelfCollisionConstraint(
        const std::string& constraint_name,
        const std::vector<int>& collision_spheres_map, // link index of each collision sphere
        const std::vector<float>& collision_spheres_radius, // radius of each collision sphere
        const std::vector<std::vector<bool>>& self_collision_enables_map
    )
    : BaseConstraint(constraint_name, false) // This constraint is not projectable.
    {
        int num_of_links = self_collision_enables_map.size();
        size_t self_collision_enables_map_bytes = (size_t)num_of_links * num_of_links * sizeof(int);
        cudaMalloc(&d_self_collision_enables_map, self_collision_enables_map_bytes);

        // Copy the self collision enables map to the device
        cudaMemcpy(d_self_collision_enables_map, boolMatrixFlatten(self_collision_enables_map).data(), self_collision_enables_map_bytes, cudaMemcpyHostToDevice);

        // determine which pairs of collision spheres should be checked for self-collision.
        std::vector<int> collision_sphere_indices_1;
        std::vector<int> collision_sphere_indices_2;
        std::vector<float> collision_distance_threshold;
        num_of_self_collision_check_ = 0;

        for (int i = 0; i < collision_spheres_map.size(); i++){
            for (int j = i + 1; j < collision_spheres_map.size(); j++){
                // check if the two spheres are not in the same link and self-collision is enabled between the two links
                if (collision_spheres_map[i] != collision_spheres_map[j] && self_collision_enables_map[collision_spheres_map[i]][collision_spheres_map[j]]){
                    collision_sphere_indices_1.push_back(i);
                    collision_sphere_indices_2.push_back(j);
                    collision_distance_threshold.push_back((collision_spheres_radius[i] + collision_spheres_radius[j]) * (collision_spheres_radius[i] + collision_spheres_radius[j])); // squared distance threshold
                    num_of_self_collision_check_++;
                }
            }
        }
        
        size_t self_collision_spheres_check_bytes = (size_t)num_of_self_collision_check_ * sizeof(int);
        size_t self_collision_check_threshold_bytes = (size_t)num_of_self_collision_check_ * sizeof(float);
        cudaMalloc(&d_collision_sphere_indices_1, self_collision_spheres_check_bytes);
        cudaMalloc(&d_collision_sphere_indices_2, self_collision_spheres_check_bytes);
        cudaMalloc(&d_collision_distance_threshold, self_collision_check_threshold_bytes);

        // Copy the collision sphere indices to the device
        cudaMemcpy(d_collision_sphere_indices_1, collision_sphere_indices_1.data(), self_collision_spheres_check_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision_sphere_indices_2, collision_sphere_indices_2.data(), self_collision_spheres_check_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision_distance_threshold, collision_distance_threshold.data(), self_collision_check_threshold_bytes, cudaMemcpyHostToDevice);
    }

    SelfCollisionConstraint::~SelfCollisionConstraint()
    {
        cudaFree(d_self_collision_enables_map);
        cudaFree(d_collision_sphere_indices_1);
        cudaFree(d_collision_sphere_indices_2);
        cudaFree(d_collision_distance_threshold);
    }

    __global__ void computeSelfCollisionCostLargeKernel(
        float* d_self_collision_spheres_pos_in_base_link, // num_of_configurations x num_of_self_collision_spheres x 3
        float* d_self_collision_spheres_radius, // num_of_self_collision_spheres
        int num_of_self_collision_spheres,
        int num_of_configurations,
        int* d_self_collision_spheres_map, // num_of_self_collision_spheres
        int num_of_robot_links,
        int* d_self_collision_enables_map, // num_of_robot_links x num_of_robot_links
        float* d_cost // num_of_configurations
    )
    {
        // Get the index of the thread
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < num_of_configurations){
            float cost = 0.0f;
            for (int i = 0; i < num_of_self_collision_spheres; i++){ // For each self collision sphere
                for (int j = i + 1; j < num_of_self_collision_spheres; j++){ // For each self collision sphere
                    // check if the two spheres are not in the same link
                    int link_i = d_self_collision_spheres_map[i];
                    int link_j = d_self_collision_spheres_map[j];
                    if ( link_i != link_j){
                        // check if two links are enabled for collision
                        if (d_self_collision_enables_map[link_i * num_of_robot_links + link_j] != 0)
                        {
                            float diff_in_x = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 0] - d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + j * 3 + 0];
                            float diff_in_y = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 1] - d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + j * 3 + 1];
                            float diff_in_z = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 2] - d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + j * 3 + 2];

                            float distance = sqrt(diff_in_x * diff_in_x + diff_in_y * diff_in_y + diff_in_z * diff_in_z); // Euclidean distance
                            float sum_of_radius = d_self_collision_spheres_radius[i] + d_self_collision_spheres_radius[j];

                            // the cost the overlap of the two spheres
                            cost += fmaxf(0.0f, sum_of_radius - distance);
                        }
                    }
                }
            }
            d_cost[idx] = cost;
        }
    }

    __global__ void computeSelfCollisionCostKernel(
        const float* __restrict__ d_self_collision_spheres_pos_in_base_link, // [num_configurations x num_spheres x 3]
        const float* __restrict__ d_self_collision_spheres_radius,            // [num_spheres]
        int num_self_collision_spheres,
        int num_configurations,
        const int* __restrict__ d_self_collision_spheres_map,                 // [num_spheres]
        int num_robot_links,
        const int* __restrict__ d_self_collision_enables_map,                 // [num_robot_links x num_robot_links]
        float* d_cost                                                         // [num_configurations x num_spheres]
    )
    {
        // Global thread index across all configurations and spheres.
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        // Ensure the thread is within the total work size.
        if (idx < num_configurations * num_self_collision_spheres)
        {
            // Calculate configuration and sphere indices.
            int config_idx = idx / num_self_collision_spheres;
            int sphere_idx = idx % num_self_collision_spheres;

            // Compute base index for this configuration.
            int config_base = config_idx * num_self_collision_spheres * 3;
            int pos_index = sphere_idx * 3;

            // Load current sphere's position.
            float cur_x = d_self_collision_spheres_pos_in_base_link[config_base + pos_index + 0];
            float cur_y = d_self_collision_spheres_pos_in_base_link[config_base + pos_index + 1];
            float cur_z = d_self_collision_spheres_pos_in_base_link[config_base + pos_index + 2];

            // Get the current sphere's link and radius.
            int current_link = d_self_collision_spheres_map[sphere_idx];
            float current_radius = d_self_collision_spheres_radius[sphere_idx];

            float cost = 0.0f;

            // Loop over all spheres for collision checks.
            for (int i = 0; i < num_self_collision_spheres; i++)
            {
                if (i == sphere_idx)
                    continue;

                int other_link = d_self_collision_spheres_map[i];
                // Only check if spheres are on different links and collisions are enabled.
                if (current_link != other_link &&
                    d_self_collision_enables_map[current_link * num_robot_links + other_link] != 0)
                {
                    int pos_index2 = i * 3;
                    float other_radius = d_self_collision_spheres_radius[i];

                    // Compute squared Euclidean distance.
                    float diff_x = cur_x - d_self_collision_spheres_pos_in_base_link[config_base + pos_index2 + 0];
                    float diff_y = cur_y - d_self_collision_spheres_pos_in_base_link[config_base + pos_index2 + 1];
                    float diff_z = cur_z - d_self_collision_spheres_pos_in_base_link[config_base + pos_index2 + 2];
                    float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                    float sum_radii = current_radius + other_radius;
                    float sum_radii_sq = sum_radii * sum_radii;

                    // Only compute sqrt when spheres overlap.
                    if (dist_sq < sum_radii_sq)
                    {
                        float distance = sqrtf(dist_sq);
                        cost += sum_radii - distance;
                    }
                }
            }

            // Write the computed cost for this sphere in this configuration.
            d_cost[idx] = cost;
        }
    }

    __global__ void sumSelfCollisionCostFastKernel(
        const float* d_cost, // num_of_configurations x num_of_elements
        int num_of_elements,
        int num_of_configurations,
        float* d_sum_cost // num_of_configurations
    )
    {
        extern __shared__ float sdata[];
        
        // Each block handles one configuration.
        int configIdx = blockIdx.x;
        if (configIdx >= num_of_configurations)
            return;

        int tid = threadIdx.x;
        int totalElements = num_of_elements;
        float sum = 0.0f;

        // Each thread sums a portion of the elements using striding
        for (int i = tid; i < totalElements; i += blockDim.x) {
            int index = configIdx * totalElements + i;
            sum += d_cost[index];
        }
        
        // Store the partial sum in shared memory
        sdata[tid] = sum;
        __syncthreads();

        // Perform reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write the result for this configuration to global memory
        if (tid == 0) {
            d_sum_cost[configIdx] = sdata[0];
        }
    }

    __global__ void sumSelfCollisionCostKernel(
        float* d_cost, // num_of_configurations x num_of_self_collision_spheres
        int num_of_self_collision_spheres,
        int num_of_configurations,
        float* d_sum_cost // num_of_configurations
    )
    {
        // Get the index of the thread
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < num_of_configurations){
            float sum_cost = 0.0f;
            for (int i = 0; i < num_of_self_collision_spheres; i++){
                sum_cost += d_cost[idx * num_of_self_collision_spheres + i];
            }
            d_sum_cost[idx] = sum_cost;
        }
    }

    void SelfCollisionConstraint::computeCostLarge(BaseStatesPtr states)
    {
        // Cast the states and space information for SingleArmSpace
        SingleArmSpaceInfoPtr space_info = std::static_pointer_cast<SingleArmSpaceInfo>(states->getSpaceInfo());
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

        // check the cost location of this constraint
        int constraint_index = getConstraintIndex(space_info);
        if (constraint_index == -1){
            // raise an error
            printf("Constraint %s is not found in the space\n", this->constraint_name.c_str());
            return;
        }

        float * d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);

        int threadsPerBlock = 256;
        int blocksPerGrid = (single_arm_states->getNumOfStates() + threadsPerBlock - 1) / threadsPerBlock;

        computeSelfCollisionCostLargeKernel<<<blocksPerGrid, threadsPerBlock>>>(
            single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(), 
            space_info->d_self_collision_spheres_radius, 
            space_info->num_of_self_collision_spheres, 
            single_arm_states->getNumOfStates(), 
            space_info->d_collision_spheres_to_link_map,
            space_info->num_of_links,
            d_self_collision_enables_map,
            d_cost_of_current_constraint
        );

        // wait for the kernel to finish
        CUDA_CHECK(cudaGetLastError()); // Check for launch errors
        CUDA_CHECK(cudaDeviceSynchronize());
    }


    void SelfCollisionConstraint::computeCost(BaseStatesPtr states)
    {
        // Cast the states and space information for SingleArmSpace
        SingleArmSpaceInfoPtr space_info = std::static_pointer_cast<SingleArmSpaceInfo>(states->getSpaceInfo());
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

        // check the cost location of this constraint
        int constraint_index = getConstraintIndex(space_info);
        if (constraint_index == -1){
            // raise an error
            printf("Constraint %s is not found in the space\n", this->constraint_name.c_str());
            return;
        }

        float * d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);

        float * d_collision_cost;
        size_t num_of_collision_pairs = (size_t)(single_arm_states->getNumOfStates()) * space_info->num_of_self_collision_spheres;
        size_t d_collision_cost_bytes = num_of_collision_pairs * sizeof(float);
        cudaMalloc(&d_collision_cost, d_collision_cost_bytes);

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_collision_pairs + threadsPerBlock - 1) / threadsPerBlock;

        // auto start_first_kernel = std::chrono::high_resolution_clock::now();

        computeSelfCollisionCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
            single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(), 
            space_info->d_self_collision_spheres_radius, 
            space_info->num_of_self_collision_spheres, 
            single_arm_states->getNumOfStates(), 
            space_info->d_collision_spheres_to_link_map,
            space_info->num_of_links,
            d_self_collision_enables_map,
            d_collision_cost
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // auto end_first_kernel = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed_seconds = end_first_kernel - start_first_kernel;
        // std::cout << "Self constraint Elapsed time for the first kernel: " << elapsed_seconds.count() << "s\n";

        blocksPerGrid = (single_arm_states->getNumOfStates() + threadsPerBlock - 1) / threadsPerBlock;

        // auto start_second_kernel = std::chrono::high_resolution_clock::now();

        sumSelfCollisionCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_collision_cost,
            space_info->num_of_self_collision_spheres,
            single_arm_states->getNumOfStates(),
            d_cost_of_current_constraint
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // auto end_second_kernel = std::chrono::high_resolution_clock::now();
        // elapsed_seconds = end_second_kernel - start_second_kernel;
        // std::cout << "Self constraint Elapsed time for the second kernel: " << elapsed_seconds.count() << "s\n";

        cudaFree(d_collision_cost);
    }

    __global__ void computeSelfCollisionCostFastKernel(
        const float* __restrict__ d_self_collision_spheres_pos_in_base_link, // num_of_configurations x num_of_self_collision_spheres x 3
        const int num_of_configurations,
        const int num_of_self_collision_spheres,
        const int num_of_self_collision_check_per_config,
        const int* __restrict__ d_collision_sphere_indices_1,
        const int* __restrict__ d_collision_sphere_indices_2,
        const float* __restrict__ d_collision_distance_threshold,
        float* d_self_collision_costs // num_of_configurations x num_of_self_collision_check_per_config
    )
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < num_of_configurations * num_of_self_collision_check_per_config)
        {
            int config_idx = idx / num_of_self_collision_check_per_config;
            int collision_check_idx = idx % num_of_self_collision_check_per_config;

            int sphere_idx_1 = d_collision_sphere_indices_1[collision_check_idx];
            int sphere_idx_2 = d_collision_sphere_indices_2[collision_check_idx];

            int base_idx = config_idx * num_of_self_collision_spheres * 3;

            // get the positions of the two spheres
            float collision_sphere_1_pos_x = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_1 * 3 + 0];
            float collision_sphere_1_pos_y = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_1 * 3 + 1];
            float collision_sphere_1_pos_z = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_1 * 3 + 2];

            float collision_sphere_2_pos_x = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_2 * 3 + 0];
            float collision_sphere_2_pos_y = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_2 * 3 + 1];
            float collision_sphere_2_pos_z = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_2 * 3 + 2];

            // compute squared distance
            float diff_in_x = collision_sphere_1_pos_x - collision_sphere_2_pos_x;
            float diff_in_y = collision_sphere_1_pos_y - collision_sphere_2_pos_y;
            float diff_in_z = collision_sphere_1_pos_z - collision_sphere_2_pos_z;

            float squared_distance = diff_in_x * diff_in_x + diff_in_y * diff_in_y + diff_in_z * diff_in_z; // squared Euclidean distance
            float collision_distance_threshold_sq = d_collision_distance_threshold[collision_check_idx];

            // the cost the overlap of the two spheres
            float self_collision_cost = 0.0f;
            if (squared_distance < collision_distance_threshold_sq){
                float sum_of_radius = sqrtf(collision_distance_threshold_sq);
                self_collision_cost = sum_of_radius - sqrtf(squared_distance);
            }

            d_self_collision_costs[idx] = self_collision_cost;
        }
    }

    // void SelfCollisionConstraint::computeCostFast(BaseStatesPtr states)
    // {
    //     // Cast the states and space information for SingleArmSpace
    //     SingleArmSpaceInfoPtr space_info = std::static_pointer_cast<SingleArmSpaceInfo>(states->getSpaceInfo());
    //     SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

    //     // check the cost location of this constraint
    //     int constraint_index = getConstraintIndex(space_info);
    //     if (constraint_index == -1){
    //         // raise an error
    //         printf("Constraint %s is not found in the space\n", this->constraint_name.c_str());
    //         return;
    //     }

    //     size_t d_self_collision_costs_bytes = (size_t)(single_arm_states->getNumOfStates()) * num_of_self_collision_check_ * sizeof(float);
    //     float * d_self_collision_costs;
    //     cudaMalloc(&d_self_collision_costs, d_self_collision_costs_bytes);

    //     // get constraint cost location
    //     float * d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);
        
    //     int threadsPerBlock = 256;
    //     int blocksPerGrid = ((size_t)(single_arm_states->getNumOfStates()) * num_of_self_collision_check_ + threadsPerBlock - 1) / threadsPerBlock;

    //     computeSelfCollisionCostFastKernel<<<blocksPerGrid, threadsPerBlock>>>(
    //         single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(), 
    //         single_arm_states->getNumOfStates(), 
    //         space_info->num_of_self_collision_spheres, 
    //         num_of_self_collision_check_,
    //         d_collision_sphere_indices_1,
    //         d_collision_sphere_indices_2,
    //         d_collision_distance_threshold,
    //         d_self_collision_costs
    //     );

    //     CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    //     CUDA_CHECK(cudaDeviceSynchronize());

    //     blocksPerGrid = single_arm_states->getNumOfStates();
    //     size_t sharedMemSize = threadsPerBlock * sizeof(float);

    //     sumSelfCollisionCostFastKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
    //         d_self_collision_costs, 
    //         num_of_self_collision_check_, 
    //         single_arm_states->getNumOfStates(), 
    //         d_cost_of_current_constraint
    //     );

    //     CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    //     CUDA_CHECK(cudaDeviceSynchronize());

    //     cudaFree(d_self_collision_costs);
    // }

    __global__ void computeAndSumSelfCollisionCostKernel(
        const float* __restrict__ d_self_collision_spheres_pos_in_base_link, // [num_configurations x num_spheres x 3]
        const int num_configurations,
        const int num_spheres,
        const int num_collision_checks_per_config,
        const int* __restrict__ d_collision_sphere_indices_1,
        const int* __restrict__ d_collision_sphere_indices_2,
        const float* __restrict__ d_collision_distance_threshold, // thresholds stored as squared values
        float* d_sum_cost // [num_configurations]
    )
    {
        int configIdx = blockIdx.x;
        if (configIdx >= num_configurations)
            return;

        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        float partialSum = 0.0f;

        // Each thread processes a subset of the collision checks using striding.
        for (int collision_check_idx = tid; collision_check_idx < num_collision_checks_per_config; collision_check_idx += blockDim.x)
        {
            int sphere_idx_1 = d_collision_sphere_indices_1[collision_check_idx];
            int sphere_idx_2 = d_collision_sphere_indices_2[collision_check_idx];
            int base_idx = configIdx * num_spheres * 3;

            // Retrieve sphere positions for this configuration.
            float pos1_x = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_1 * 3 + 0];
            float pos1_y = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_1 * 3 + 1];
            float pos1_z = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_1 * 3 + 2];

            float pos2_x = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_2 * 3 + 0];
            float pos2_y = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_2 * 3 + 1];
            float pos2_z = d_self_collision_spheres_pos_in_base_link[base_idx + sphere_idx_2 * 3 + 2];

            // Compute squared Euclidean distance.
            float dx = pos1_x - pos2_x;
            float dy = pos1_y - pos2_y;
            float dz = pos1_z - pos2_z;
            float squared_distance = dx * dx + dy * dy + dz * dz;

            // Get the squared collision threshold.
            float collision_distance_threshold_sq = d_collision_distance_threshold[collision_check_idx];

            // If the spheres are closer than the threshold, compute the cost.
            float cost = 0.0f;
            if (squared_distance < collision_distance_threshold_sq)
            {
                float sum_of_radius = sqrtf(collision_distance_threshold_sq);
                cost = sum_of_radius - sqrtf(squared_distance);
            }
            partialSum += cost;
        }

        // Store the partial sum for this thread in shared memory.
        sdata[tid] = partialSum;
        __syncthreads();

        // Parallel reduction in shared memory.
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write the final reduced sum to global memory.
        if (tid == 0)
        {
            d_sum_cost[configIdx] = sdata[0];
        }
    }

    // Host function that computes the self-collision cost using the combined kernel.
    void SelfCollisionConstraint::computeCostFast(BaseStatesPtr states) {
        // Cast the states and space information for SingleArmSpace
        SingleArmSpaceInfoPtr space_info = std::static_pointer_cast<SingleArmSpaceInfo>(states->getSpaceInfo());
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

        // check the cost location of this constraint
        int constraint_index = getConstraintIndex(space_info);
        if (constraint_index == -1){
            // raise an error
            printf("Constraint %s is not found in the space\n", this->constraint_name.c_str());
            return;
        }

        // Get the pointer to the cost location for the current constraint.
        float* d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);

        // Launch parameters.
        int threadsPerBlock = 256;
        int numConfigs = single_arm_states->getNumOfStates();
        int blocksPerGrid = numConfigs;  // one block per configuration
        size_t sharedMemSize = threadsPerBlock * sizeof(float);

        // Launch the combined kernel.
        computeAndSumSelfCollisionCostKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(),
            numConfigs,
            space_info->num_of_self_collision_spheres,
            num_of_self_collision_check_,
            d_collision_sphere_indices_1,
            d_collision_sphere_indices_2,
            d_collision_distance_threshold,
            d_cost_of_current_constraint
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
} // namespace CUDAMPLib
#include <constraints/SelfCollisionConstraint.h>

#include <chrono>

namespace CUDAMPLib{

    SelfCollisionConstraint::SelfCollisionConstraint(
        const std::string& constraint_name,
        const std::vector<std::vector<bool>>& self_collision_enables_map
    )
    : BaseConstraint(constraint_name)
    {
        int num_of_links = self_collision_enables_map.size();
        size_t self_collision_enables_map_bytes = (size_t)num_of_links * num_of_links * sizeof(int);
        cudaMalloc(&d_self_collision_enables_map, self_collision_enables_map_bytes);

        // Copy the self collision enables map to the device
        cudaMemcpy(d_self_collision_enables_map, boolMatrixFlatten(self_collision_enables_map).data(), self_collision_enables_map_bytes, cudaMemcpyHostToDevice);
    }

    SelfCollisionConstraint::~SelfCollisionConstraint()
    {
        cudaFree(d_self_collision_enables_map);
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
                        if (d_self_collision_enables_map[link_i * num_of_robot_links + link_j] == 1)
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
                    d_self_collision_enables_map[current_link * num_robot_links + other_link] == 1)
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

    // __global__ void computeSelfCollisionCostKernel(
    //     const float* __restrict__ d_self_collision_spheres_pos_in_base_link, // num_of_configurations x num_of_self_collision_spheres x 3
    //     const float* __restrict__ d_self_collision_spheres_radius, // num_of_self_collision_spheres
    //     int num_of_self_collision_spheres,
    //     int num_of_configurations,
    //     const int* __restrict__ d_self_collision_spheres_map, // num_of_self_collision_spheres
    //     int num_of_robot_links,
    //     const int* __restrict__ d_self_collision_enables_map, // num_of_robot_links x num_of_robot_links
    //     float* d_cost // num_of_configurations x num_of_self_collision_spheres
    // )
    // {
    //     int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //     if (idx < num_of_configurations * num_of_self_collision_spheres)
    //     {
    //         int config_idx = idx / num_of_self_collision_spheres;
    //         int sphere_idx = idx % num_of_self_collision_spheres;

    //         float cost = 0.0f;
    //         for (int i = 0; i < num_of_self_collision_spheres; i++){ // For each self collision sphere
    //             if (i != sphere_idx){
    //                 // check if the two spheres are not in the same link
    //                 int link_i = d_self_collision_spheres_map[sphere_idx];
    //                 int link_j = d_self_collision_spheres_map[i];
    //                 if ( link_i != link_j){
    //                     // check if two links are enabled for collision
    //                     if (d_self_collision_enables_map[link_i * num_of_robot_links + link_j] == 1)
    //                     {
    //                         float diff_in_x = d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + sphere_idx * 3 + 0] - d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + i * 3 + 0];
    //                         float diff_in_y = d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + sphere_idx * 3 + 1] - d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + i * 3 + 1];
    //                         float diff_in_z = d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + sphere_idx * 3 + 2] - d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + i * 3 + 2];

    //                         float distance = sqrtf(diff_in_x * diff_in_x + diff_in_y * diff_in_y + diff_in_z * diff_in_z); // Euclidean distance
    //                         float sum_of_radius = d_self_collision_spheres_radius[sphere_idx] + d_self_collision_spheres_radius[i];

    //                         // the cost the overlap of the two spheres
    //                         cost += fmaxf(0.0f, sum_of_radius - distance);
    //                     }
    //                 }
    //             }
    //         }
    //         d_cost[idx] = cost;
    //     }
    // }

    /**
        @brief Compute the cost between each pair of self collision spheres for each configuration.
        @param d_self_collision_spheres_pos_in_base_link [num_of_configurations x num_of_self_collision_spheres x 3] The positions of self collision spheres in the base link frame.
        @param d_self_collision_spheres_radius [num_of_self_collision_spheres] The radius of each self collision sphere.
        @param num_of_self_collision_spheres The number of self collision spheres.
        @param num_of_configurations The number of configurations.
        @param d_self_collision_spheres_map [num_of_self_collision_spheres] The link index of each self collision sphere.
        @param num_of_robot_links The number of links in the robot.
        @param d_self_collision_enables_map [num_of_robot_links x num_of_robot_links] The map of enabled self collision between links.
        @param d_cost [num_of_configurations x num_of_self_collision_spheres x num_of_self_collision_spheres] The cost between each pair of self collision spheres for each configuration.
     */
    __global__ void computeSelfCollisionCostFastKernel(
        const float* __restrict__ d_self_collision_spheres_pos_in_base_link, // num_of_configurations x num_of_self_collision_spheres x 3
        const float* __restrict__ d_self_collision_spheres_radius, // num_of_self_collision_spheres
        int num_of_self_collision_spheres,
        int num_of_configurations,
        const int* __restrict__ d_self_collision_spheres_map, // num_of_self_collision_spheres
        int num_of_robot_links,
        const int* __restrict__ d_self_collision_enables_map, // num_of_robot_links x num_of_robot_links
        float* d_cost // num_of_configurations x num_of_self_collision_spheres x num_of_self_collision_spheres
    )
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < num_of_configurations * num_of_self_collision_spheres * num_of_self_collision_spheres)
        {
            int config_idx = idx / ( num_of_self_collision_spheres * num_of_self_collision_spheres);
            int sphere_idx = (idx / num_of_self_collision_spheres) % num_of_self_collision_spheres;
            int other_sphere_idx = idx % num_of_self_collision_spheres;

            float cost = 0.0f;

            // check if the two spheres are not in the same link
            int link_i = d_self_collision_spheres_map[sphere_idx];
            int link_j = d_self_collision_spheres_map[other_sphere_idx];

            if ( link_i != link_j){
                // check if two links are enabled for collision
                if (d_self_collision_enables_map[link_i * num_of_robot_links + link_j] == 1)
                {
                    float diff_in_x = d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + sphere_idx * 3 + 0] - d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + other_sphere_idx * 3 + 0];
                    float diff_in_y = d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + sphere_idx * 3 + 1] - d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + other_sphere_idx * 3 + 1];
                    float diff_in_z = d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + sphere_idx * 3 + 2] - d_self_collision_spheres_pos_in_base_link[config_idx * num_of_self_collision_spheres * 3 + other_sphere_idx * 3 + 2];

                    float distance = sqrtf(diff_in_x * diff_in_x + diff_in_y * diff_in_y + diff_in_z * diff_in_z); // Euclidean distance
                    float sum_of_radius = d_self_collision_spheres_radius[sphere_idx] + d_self_collision_spheres_radius[other_sphere_idx];

                    // the cost the overlap of the two spheres
                    cost = fmaxf(0.0f, sum_of_radius - distance);
                }
            }

            d_cost[idx] = cost;
        }
    }

    __global__ void sumSelfCollisionCostFastKernel(
        const float* d_cost, // num_of_configurations x num_of_self_collision_spheres x num_of_self_collision_spheres
        int num_of_self_collision_spheres,
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
        int totalElements = num_of_self_collision_spheres * num_of_self_collision_spheres;
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

    void SelfCollisionConstraint::computeCostFast(BaseStatesPtr states)
    {
        // computeCost(states);

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

        size_t num_of_collision_pairs = (size_t)(single_arm_states->getNumOfStates()) * space_info->num_of_self_collision_spheres * space_info->num_of_self_collision_spheres;
        size_t d_collision_cost_bytes = num_of_collision_pairs * sizeof(float);

        size_t free_byte, total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);

        // if memory is not enough, use the original method
        if (d_collision_cost_bytes > 0.1 * free_byte){
            // printf("Not enough memory, use the original method\n");
            computeCost(states);
            return;
        }

        float * d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);
        float * d_collision_cost;
        cudaMalloc(&d_collision_cost, d_collision_cost_bytes);
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_collision_pairs + threadsPerBlock - 1) / threadsPerBlock;

        // printf("d_collision_cost_bytes: %zu, free_byte: %zu, total_byte: %zu\n", d_collision_cost_bytes, free_byte, total_byte);

        // auto start_first_kernel = std::chrono::high_resolution_clock::now();

        computeSelfCollisionCostFastKernel<<<blocksPerGrid, threadsPerBlock>>>(
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

        blocksPerGrid = single_arm_states->getNumOfStates();
        size_t sharedMemSize = threadsPerBlock * sizeof(float);

        // auto start_second_kernel = std::chrono::high_resolution_clock::now();

        sumSelfCollisionCostFastKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_collision_cost, 
            space_info->num_of_self_collision_spheres, 
            single_arm_states->getNumOfStates(), 
            d_cost_of_current_constraint
        );

        CUDA_CHECK(cudaGetLastError()); // Check for launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        // auto end_second_kernel = std::chrono::high_resolution_clock::now();
        // elapsed_seconds = end_second_kernel - start_second_kernel;
        // std::cout << "Self constraint Elapsed time for the second kernel: " << elapsed_seconds.count() << "s\n";

        cudaFree(d_collision_cost);
    }
} // namespace CUDAMPLib
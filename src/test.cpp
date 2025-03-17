__global__ void computeCollisionCostFastKernel(
    const float* __restrict__ d_self_collision_spheres_pos_in_base_link, // [num_configurations x num_self_collision_spheres x 3]
    const float* __restrict__ d_self_collision_spheres_radius,            // [num_self_collision_spheres]
    int num_of_self_collision_spheres,
    int num_of_configurations,
    const float* __restrict__ d_obstacle_sphere_pos_in_base_link,         // [num_of_obstacle_collision_spheres x 3]
    const float* __restrict__ d_obstacle_sphere_radius,                   // [num_of_obstacle_collision_spheres]
    int num_of_obstacle_collision_spheres,
    float* d_cost                                                         // [num_configurations x num_of_self_collision_spheres x num_of_obstacle_collision_spheres]
)
{
    // Global thread index: one thread per self-collision sphere per configuration.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_of_configurations * num_of_self_collision_spheres * num_of_obstacle_collision_spheres)
    {
        float cost = 0.0f;

        // Determine configuration, sphere, and obstacle indices.
        int configuration_index = idx / (num_of_self_collision_spheres * num_of_obstacle_collision_spheres);
        int self_collision_sphere_index = (idx / num_of_obstacle_collision_spheres) % num_of_self_collision_spheres;
        int obstacle_index = idx % num_of_obstacle_collision_spheres;

        // Compute the base index for this configuration.
        int config_base = configuration_index * num_of_self_collision_spheres * 3;
        int pos_index = config_base + self_collision_sphere_index * 3;

        // Load current sphere's position and radius.
        float self_x = d_self_collision_spheres_pos_in_base_link[pos_index + 0];
        float self_y = d_self_collision_spheres_pos_in_base_link[pos_index + 1];
        float self_z = d_self_collision_spheres_pos_in_base_link[pos_index + 2];
        float self_radius = d_self_collision_spheres_radius[self_collision_sphere_index];

        // Load obstacle sphere's position and radius.
        int obs_index = obstacle_index * 3;
        float obs_x = d_obstacle_sphere_pos_in_base_link[obs_index + 0];
        float obs_y = d_obstacle_sphere_pos_in_base_link[obs_index + 1];
        float obs_z = d_obstacle_sphere_pos_in_base_link[obs_index + 2];
        float obs_radius = d_obstacle_sphere_radius[obstacle_index];

        // Compute distance and cost.
        float diff_x = self_x - obs_x;
        float diff_y = self_y - obs_y;
        float diff_z = self_z - obs_z;
        float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

        float sum_radii = self_radius + obs_radius;
        float sum_radii_sq = sum_radii * sum_radii;

        // Only compute the square root if spheres are overlapping.
        if (dist_sq < sum_radii_sq)
        {
            float distance = sqrtf(dist_sq);
            cost = fmaxf(0.0f, sum_radii - distance);
        }

        // Write the computed cost back to global memory.
        d_cost[idx] = cost;
    }
}

/**
    * @brief Kernel to sum the collision cost for each configuration with reduction.
    */
__global__ void sumCollisionCostFastKernel(
    const float* d_collision_cost,  // num_of_states x num_of_self_collision_spheres x num_of_obstacle_collision_spheres
    int num_of_states,
    int num_of_self_collision_spheres,
    int num_of_obstacle_collision_spheres,
    float* d_cost                   // num_of_states
)
{
    extern __shared__ float sdata[];

    // Each block handles one state.
    int state_index = blockIdx.x;
    if (state_index >= num_of_states)
        return;

    int tid = threadIdx.x;
    int totalElements = num_of_self_collision_spheres * num_of_obstacle_collision_spheres;
    float sum = 0.0f;

    // Each thread sums over a strided portion of the flattened cost matrix.
    for (int i = tid; i < totalElements; i += blockDim.x) {
        int index = state_index * totalElements + i;
        sum += d_collision_cost[index];
    }
    
    // Store the partial sum in shared memory.
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this state to global memory.
    if (tid == 0) {
        d_cost[state_index] = sdata[0];
    }
}

void EnvConstraint::computeCostFast(BaseStatesPtr states)
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

    size_t num_of_env_collision_check = (size_t)(single_arm_states->getNumOfStates()) * space_info->num_of_self_collision_spheres * num_of_env_collision_spheres;
    size_t d_collision_cost_bytes = num_of_env_collision_check * sizeof(float);

    float * d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);
    float * d_collision_cost;
    
    cudaMalloc(&d_collision_cost, d_collision_cost_bytes);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_of_env_collision_check + threadsPerBlock - 1) / threadsPerBlock;

    auto start_first_kernel = std::chrono::high_resolution_clock::now();

    computeCollisionCostFastKernel<<<blocksPerGrid, threadsPerBlock>>>(
        single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(), 
        space_info->d_self_collision_spheres_radius, 
        space_info->num_of_self_collision_spheres, 
        single_arm_states->getNumOfStates(), 
        d_env_collision_spheres_pos_in_base_link, 
        d_env_collision_spheres_radius, 
        num_of_env_collision_spheres, 
        d_collision_cost 
    );
    
    // wait for the kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // sum the collision cost
    blocksPerGrid = single_arm_states->getNumOfStates();

    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    sumCollisionCostFastKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_collision_cost,
        single_arm_states->getNumOfStates(),
        space_info->num_of_self_collision_spheres,
        num_of_env_collision_spheres,
        d_cost_of_current_constraint
    );

    // wait for the kernel to finish
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_collision_cost);
}
#include <constraints/SelfCollisionConstraint.h>

#include <chrono>

namespace CUDAMPLib{

    SelfCollisionConstraint::SelfCollisionConstraint(
        const std::string& constraint_name,
        const std::vector<int>& self_collision_spheres_map, // link index of each collision sphere
        const std::vector<float>& self_collision_spheres_radius, // radius of each collision sphere
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
        collision_sphere_indices_1.clear();
        collision_sphere_indices_2.clear();
        collision_distance_threshold.clear();
        num_of_self_collision_check_ = 0;

        for (size_t i = 0; i < self_collision_spheres_map.size(); i++){
            for (size_t j = i + 1; j < self_collision_spheres_map.size(); j++){
                // check if the two spheres are not in the same link and self-collision is enabled between the two links
                if (self_collision_spheres_map[i] != self_collision_spheres_map[j] && self_collision_enables_map[self_collision_spheres_map[i]][self_collision_spheres_map[j]]){
                    collision_sphere_indices_1.push_back(i);
                    collision_sphere_indices_2.push_back(j);
                    collision_distance_threshold.push_back((self_collision_spheres_radius[i] + self_collision_spheres_radius[j]) * (self_collision_spheres_radius[i] + self_collision_spheres_radius[j])); // squared distance threshold
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
    void SelfCollisionConstraint::computeCost(BaseStatesPtr states) {
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

        // CUDA_CHECK(cudaGetLastError());
        // CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::string SelfCollisionConstraint::generateCheckConstraintCode()
    {
        std::string source_code;
        source_code += "// SelfCollisionConstraint check function\n";
        source_code += "__device__ __forceinline__ void checkSelfCollisionConstraint(bool * should_skip, float * self_collision_sphere_pos){\n";
        source_code += "    float dx = 0.0f;\n";
        source_code += "    float dy = 0.0f;\n";
        source_code += "    float dz = 0.0f;\n";
        source_code += "    float squared_distance = 0.0f;\n";
        std::cout << "=========== num_of_self_collision_check_: " << num_of_self_collision_check_ << std::endl;
        for (int i = 0; i < num_of_self_collision_check_; i++)
        {
            source_code += "    dx = self_collision_sphere_pos[3 * " + std::to_string(collision_sphere_indices_1[i]) + "] - self_collision_sphere_pos[3 * " + std::to_string(collision_sphere_indices_2[i]) + "];\n";
            source_code += "    dy = self_collision_sphere_pos[3 * " + std::to_string(collision_sphere_indices_1[i]) + " + 1] - self_collision_sphere_pos[3 * " + std::to_string(collision_sphere_indices_2[i]) + " + 1];\n";
            source_code += "    dz = self_collision_sphere_pos[3 * " + std::to_string(collision_sphere_indices_1[i]) + " + 2] - self_collision_sphere_pos[3 * " + std::to_string(collision_sphere_indices_2[i]) + " + 2];\n";
            source_code += "    squared_distance = dx * dx + dy * dy + dz * dz;\n";
            source_code += "    if (squared_distance < " + std::to_string(collision_distance_threshold[i]) + "){\n";
            source_code += "        *should_skip = true;\n";
            source_code += "    }\n";
            source_code += "    if (*should_skip == true){\n";
            source_code += "        return;\n";
            source_code += "    }\n";
        }

        source_code += "}\n";
        return source_code;
    }

    std::string SelfCollisionConstraint::generateLaunchCheckConstraintCode()
    {
        std::string source_code;
        source_code += "        // Launch SelfCollisionConstraint check function\n";
        // source_code += "        checkSelfCollisionConstraint(&should_skip, self_collision_spheres_pos_in_base);\n";
        // source_code += "        __syncthreads();\n";
        // source_code += "        if (should_skip == true){\n";
        // source_code += "            continue;\n";
        // source_code += "        }\n";
        return source_code;
    }
} // namespace CUDAMPLib
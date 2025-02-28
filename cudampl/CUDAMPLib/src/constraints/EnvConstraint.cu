#include <constraints/EnvConstraint.h>

#include <chrono>

namespace CUDAMPLib{

    EnvConstraint::EnvConstraint(
        const std::string& constraint_name,
        const std::vector<std::vector<float>>& env_collision_spheres_pos,
        const std::vector<float>& env_collision_spheres_radius
    )
    : BaseConstraint(constraint_name)
    {
        // Prepare the cuda memory for the collision cost
        num_of_env_collision_spheres = env_collision_spheres_pos.size();

        // Allocate memory for the environment collision spheres
        size_t env_collision_spheres_pos_bytes = (size_t)num_of_env_collision_spheres * sizeof(float) * 3;
        size_t env_collision_spheres_radius_bytes = (size_t)num_of_env_collision_spheres * sizeof(float);

        cudaMalloc(&d_env_collision_spheres_pos_in_base_link, env_collision_spheres_pos_bytes);
        cudaMalloc(&d_env_collision_spheres_radius, env_collision_spheres_radius_bytes);

        // Copy the environment collision spheres to the device
        cudaMemcpy(d_env_collision_spheres_pos_in_base_link, floatVectorFlatten(env_collision_spheres_pos).data(), env_collision_spheres_pos_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_env_collision_spheres_radius, env_collision_spheres_radius.data(), env_collision_spheres_radius_bytes, cudaMemcpyHostToDevice);
    }

    EnvConstraint::~EnvConstraint()
    {
        cudaFree(d_env_collision_spheres_pos_in_base_link);
        cudaFree(d_env_collision_spheres_radius);
    }

    __global__ void computeCollisionCostLargeKernel(
        float* d_self_collision_spheres_pos_in_base_link, // num_of_configurations x num_of_self_collision_spheres x 3
        float* d_self_collision_spheres_radius, // num_of_self_collision_spheres
        int num_of_self_collision_spheres,
        int num_of_configurations,
        float* d_obstacle_sphere_pos_in_base_link, // num_of_obstacle_spheres x 3
        float* d_obstacle_sphere_radius, // num_of_obstacle_spheres
        int num_of_obstacle_collision_spheres,
        float* d_cost // num_of_configurations
    )
    {
        // Get the index of the thread
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < num_of_configurations){
            float cost = 0.0f;
            for (int i = 0; i < num_of_self_collision_spheres; i++){ // For each self collision sphere
                for (int j = 0; j < num_of_obstacle_collision_spheres; j++){ // For each obstacle sphere

                    float diff_in_x = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 0] - d_obstacle_sphere_pos_in_base_link[j * 3 + 0];
                    float diff_in_y = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 1] - d_obstacle_sphere_pos_in_base_link[j * 3 + 1];
                    float diff_in_z = d_self_collision_spheres_pos_in_base_link[idx * num_of_self_collision_spheres * 3 + i * 3 + 2] - d_obstacle_sphere_pos_in_base_link[j * 3 + 2];

                    float distance = sqrt(diff_in_x * diff_in_x + diff_in_y * diff_in_y + diff_in_z * diff_in_z); // Euclidean distance
                    float sum_of_radius = d_self_collision_spheres_radius[i] + d_obstacle_sphere_radius[j];

                    // the cost the overlap of the two spheres
                    cost += fmaxf(0.0f, sum_of_radius - distance);
                }
            }
            d_cost[idx] = cost;
        }
    }

    // __global__ void computeCollisionCostKernel(
    //     float* d_self_collision_spheres_pos_in_base_link, // num_of_configurations x num_of_self_collision_spheres x 3
    //     float* d_self_collision_spheres_radius, // num_of_self_collision_spheres
    //     int num_of_self_collision_spheres,
    //     int num_of_configurations,
    //     float* d_obstacle_sphere_pos_in_base_link, // num_of_obstacle_spheres x 3
    //     float* d_obstacle_sphere_radius, // num_of_obstacle_spheres
    //     int num_of_obstacle_collision_spheres,
    //     float* d_cost // num_of_configurations x num_of_self_collision_spheres
    // )
    // {
    //     // Get the index of the thread
    //     int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //     if (idx < num_of_configurations * num_of_self_collision_spheres)
    //     {
    //         int configuration_index = idx / num_of_self_collision_spheres;
    //         int self_collision_sphere_index = idx % num_of_self_collision_spheres;

    //         float cost = 0.0f;

    //         for (int j = 0; j < num_of_obstacle_collision_spheres; j++)
    //         {
    //             float diff_in_x = d_self_collision_spheres_pos_in_base_link[configuration_index * num_of_self_collision_spheres * 3 + self_collision_sphere_index * 3 + 0] - d_obstacle_sphere_pos_in_base_link[j * 3 + 0];
    //             float diff_in_y = d_self_collision_spheres_pos_in_base_link[configuration_index * num_of_self_collision_spheres * 3 + self_collision_sphere_index * 3 + 1] - d_obstacle_sphere_pos_in_base_link[j * 3 + 1];
    //             float diff_in_z = d_self_collision_spheres_pos_in_base_link[configuration_index * num_of_self_collision_spheres * 3 + self_collision_sphere_index * 3 + 2] - d_obstacle_sphere_pos_in_base_link[j * 3 + 2];

    //             float distance = sqrt(diff_in_x * diff_in_x + diff_in_y * diff_in_y + diff_in_z * diff_in_z); // Euclidean distance
    //             float sum_of_radius = d_self_collision_spheres_radius[self_collision_sphere_index] + d_obstacle_sphere_radius[j];

    //             // the cost the overlap of the two spheres
    //             cost += fmaxf(0.0f, sum_of_radius - distance);
    //         }
    //         d_cost[idx] = cost;
    //     }
    // }

    __global__ void computeCollisionCostKernel(
        const float* __restrict__ d_self_collision_spheres_pos_in_base_link, // [num_configurations x num_self_collision_spheres x 3]
        const float* __restrict__ d_self_collision_spheres_radius,            // [num_self_collision_spheres]
        int num_of_self_collision_spheres,
        int num_of_configurations,
        const float* __restrict__ d_obstacle_sphere_pos_in_base_link,         // [num_of_obstacle_collision_spheres x 3]
        const float* __restrict__ d_obstacle_sphere_radius,                   // [num_of_obstacle_collision_spheres]
        int num_of_obstacle_collision_spheres,
        float* d_cost                                                         // [num_configurations x num_of_self_collision_spheres]
    )
    {
        // Global thread index: one thread per self-collision sphere per configuration.
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < num_of_configurations * num_of_self_collision_spheres)
        {
            // Determine configuration and sphere indices.
            int configuration_index = idx / num_of_self_collision_spheres;
            int self_collision_sphere_index = idx % num_of_self_collision_spheres;

            // Compute the base index for this configuration.
            int config_base = configuration_index * num_of_self_collision_spheres * 3;
            int pos_index = config_base + self_collision_sphere_index * 3;

            // Load current sphere's position and radius.
            float self_x = d_self_collision_spheres_pos_in_base_link[pos_index + 0];
            float self_y = d_self_collision_spheres_pos_in_base_link[pos_index + 1];
            float self_z = d_self_collision_spheres_pos_in_base_link[pos_index + 2];
            float self_radius = d_self_collision_spheres_radius[self_collision_sphere_index];

            float cost = 0.0f;

            // Loop over all obstacle spheres.
            for (int j = 0; j < num_of_obstacle_collision_spheres; j++)
            {
                int obs_index = j * 3;
                float obs_x = d_obstacle_sphere_pos_in_base_link[obs_index + 0];
                float obs_y = d_obstacle_sphere_pos_in_base_link[obs_index + 1];
                float obs_z = d_obstacle_sphere_pos_in_base_link[obs_index + 2];
                float obs_radius = d_obstacle_sphere_radius[j];

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
                    cost += fmaxf(0.0f, sum_radii - distance);
                }
            }

            // Write the computed cost back to global memory.
            d_cost[idx] = cost;
        }
    }

    __global__ void sumCollisionCostKernel(
        float* d_collision_cost,
        int num_of_states,
        int num_of_self_collision_spheres,
        float* d_cost
    )
    {
        int state_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (state_index < num_of_states){
            float cost = 0.0f;
            for (int i = 0; i < num_of_self_collision_spheres; i++){
                cost += d_collision_cost[state_index * num_of_self_collision_spheres + i];
            }
            d_cost[state_index] = cost;
        }
    }

    void EnvConstraint::computeCostLarge(BaseStatesPtr states)
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

        computeCollisionCostLargeKernel<<<blocksPerGrid, threadsPerBlock>>>(
            single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(), 
            space_info->d_self_collision_spheres_radius, 
            space_info->num_of_self_collision_spheres, 
            single_arm_states->getNumOfStates(), 
            d_env_collision_spheres_pos_in_base_link, 
            d_env_collision_spheres_radius, 
            num_of_env_collision_spheres, 
            d_cost_of_current_constraint 
        );

        cudaDeviceSynchronize();
    }

    void EnvConstraint::computeCost(BaseStatesPtr states)
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

        // allocate memory for 3d collision cost, num_of_configurations x num_of_self_collision_spheres x num_of_obstacle_collision_spheres
        float * d_collision_cost;
        size_t num_of_env_collision_check = (size_t)(single_arm_states->getNumOfStates()) * space_info->num_of_self_collision_spheres;
        size_t d_collision_cost_bytes = num_of_env_collision_check * sizeof(float);
        cudaMalloc(&d_collision_cost, d_collision_cost_bytes);

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_env_collision_check + threadsPerBlock - 1) / threadsPerBlock;

        // auto start_first_kernel = std::chrono::high_resolution_clock::now();

        computeCollisionCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
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
        blocksPerGrid = (single_arm_states->getNumOfStates() + threadsPerBlock - 1) / threadsPerBlock;

        // auto start_second_kernel = std::chrono::high_resolution_clock::now();
        sumCollisionCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_collision_cost,
            single_arm_states->getNumOfStates(),
            space_info->num_of_self_collision_spheres,
            d_cost_of_current_constraint
        );

        // wait for the kernel to finish
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_collision_cost);
    }

    void EnvConstraint::computeCostFast(BaseStatesPtr state)
    {
        computeCost(state); // to do: implement the fast version
    }
} // namespace CUDAMPLib
#include <constraints/EnvConstraintSphere.h>

#include <chrono>

namespace CUDAMPLib{

    EnvConstraintSphere::EnvConstraintSphere(
        const std::string& constraint_name,
        const std::vector<std::vector<float>>& env_collision_spheres_pos,
        const std::vector<float>& env_collision_spheres_radius
    )
    : BaseConstraint(constraint_name, false) // This constraint is not projectable.
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

    EnvConstraintSphere::~EnvConstraintSphere()
    {
        cudaFree(d_env_collision_spheres_pos_in_base_link);
        cudaFree(d_env_collision_spheres_radius);
    }

    __global__ void computeAndSumCollisionCostKernel(
        const float* __restrict__ d_self_collision_spheres_pos_in_base_link, // [num_configurations x num_self_collision_spheres x 3]
        const float* __restrict__ d_self_collision_spheres_radius,            // [num_self_collision_spheres]
        int num_of_self_collision_spheres,
        int num_of_configurations,
        const float* __restrict__ d_obstacle_sphere_pos_in_base_link,         // [num_of_obstacle_collision_spheres x 3]
        const float* __restrict__ d_obstacle_sphere_radius,                   // [num_of_obstacle_collision_spheres]
        int num_of_obstacle_collision_spheres,
        float* d_cost                                                         // [num_configurations]
    )
    {
        extern __shared__ float sdata[];

        // Each block handles one configuration (state)
        int state_index = blockIdx.x;
        if (state_index >= num_of_configurations)
            return;

        int tid = threadIdx.x;
        int totalPairs = num_of_self_collision_spheres * num_of_obstacle_collision_spheres;
        float localSum = 0.0f;

        // Process pairs in a strided loop across all self/obstacle sphere pairs.
        for (int pairIdx = tid; pairIdx < totalPairs; pairIdx += blockDim.x)
        {
            // Decode indices.
            int self_idx = pairIdx / num_of_obstacle_collision_spheres;
            int obs_idx  = pairIdx % num_of_obstacle_collision_spheres;

            // Compute the base index for the self-collision sphere position for this configuration.
            int pos_index = state_index * num_of_self_collision_spheres * 3 + self_idx * 3;
            float self_x = d_self_collision_spheres_pos_in_base_link[pos_index + 0];
            float self_y = d_self_collision_spheres_pos_in_base_link[pos_index + 1];
            float self_z = d_self_collision_spheres_pos_in_base_link[pos_index + 2];
            float self_radius = d_self_collision_spheres_radius[self_idx];

            // Load the obstacle sphere's position and radius.
            int obs_pos_index = obs_idx * 3;
            float obs_x = d_obstacle_sphere_pos_in_base_link[obs_pos_index + 0];
            float obs_y = d_obstacle_sphere_pos_in_base_link[obs_pos_index + 1];
            float obs_z = d_obstacle_sphere_pos_in_base_link[obs_pos_index + 2];
            float obs_radius = d_obstacle_sphere_radius[obs_idx];

            // Compute squared distance between sphere centers.
            float diff_x = self_x - obs_x;
            float diff_y = self_y - obs_y;
            float diff_z = self_z - obs_z;
            float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

            float sum_radii = self_radius + obs_radius;
            float sum_radii_sq = sum_radii * sum_radii;
            float cost = 0.0f;

            // Only compute the square root if spheres overlap.
            if (dist_sq < sum_radii_sq)
            {
                float distance = sqrtf(dist_sq);
                cost = fmaxf(0.0f, sum_radii - distance);
            }

            localSum += cost;
        }

        // Store the partial sum in shared memory.
        sdata[tid] = localSum;
        __syncthreads();

        // Reduction in shared memory.
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write the final summed cost for this configuration.
        if (tid == 0)
        {
            d_cost[state_index] = sdata[0];
        }
    }

    void EnvConstraintSphere::computeCost(BaseStatesPtr states)
    {
        // Cast the state and space information.
        SingleArmSpaceInfoPtr space_info = std::static_pointer_cast<SingleArmSpaceInfo>(states->getSpaceInfo());
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

        // Retrieve the cost location for this constraint.
        int constraint_index = getConstraintIndex(space_info);
        if (constraint_index == -1)
        {
            printf("Constraint %s is not found in the space\n", this->constraint_name.c_str());
            return;
        }

        // Pointer to the cost buffer for the current constraint.
        float* d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);

        // Setup kernel launch parameters.
        int threadsPerBlock = 256;
        int blocksPerGrid = single_arm_states->getNumOfStates(); // one block per configuration
        size_t sharedMemSize = threadsPerBlock * sizeof(float);

        // Launch the combined kernel.
        computeAndSumCollisionCostKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(),
            space_info->d_self_collision_spheres_radius,
            space_info->num_of_self_collision_spheres,
            single_arm_states->getNumOfStates(),
            d_env_collision_spheres_pos_in_base_link,  // assumed to be defined/available
            d_env_collision_spheres_radius,            // assumed to be defined/available
            num_of_env_collision_spheres,              // assumed to be defined/available
            d_cost_of_current_constraint
        );
    }

    std::string EnvConstraintSphere::generateCheckConstraintCode()
    {
        return "// EnvConstraintSphere check function\n";
    }

    std::string EnvConstraintSphere::generateLaunchCheckConstraintCode()
    {
        return "// Launch EnvConstraintSphere check function\n";
    }
} // namespace CUDAMPLib
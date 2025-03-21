#include <constraints/EnvConstraintCuboid.h>
#include <chrono>
#include <cstdio> // for printf
#include <vector>

namespace CUDAMPLib {

    // Helper function: computes the inverse pose matrix for a cuboid.
    // Here we assume:
    // - 'pos' is a 3-element vector [x, y, z] (cuboid position in base_link).
    // - 'orientation' is a 9-element row-major 3x3 rotation matrix.
    // The inverse pose is computed as T⁻¹ = [ Rᵀ, -Rᵀ * t; 0 0 0 1 ].
    std::vector<float> computeInversePoseMatrix(const std::vector<float>& pos, const std::vector<float>& orientation) {
        // Ensure pos has 3 and orientation has 9 elements.
        std::vector<float> inv(16, 0.0f);
        // Extract the rotation matrix R from orientation (row-major)
        float r00 = orientation[0], r01 = orientation[1], r02 = orientation[2];
        float r10 = orientation[3], r11 = orientation[4], r12 = orientation[5];
        float r20 = orientation[6], r21 = orientation[7], r22 = orientation[8];
        // Compute the transpose Rᵀ
        float t00 = r00, t01 = r10, t02 = r20;
        float t10 = r01, t11 = r11, t12 = r21;
        float t20 = r02, t21 = r12, t22 = r22;
        // Set the rotation part in the inverse matrix
        inv[0] = t00; inv[1] = t01; inv[2]  = t02;
        inv[4] = t10; inv[5] = t11; inv[6]  = t12;
        inv[8] = t20; inv[9] = t21; inv[10] = t22;
        // Compute the translation part: -Rᵀ * t
        inv[3]  = - (t00 * pos[0] + t01 * pos[1] + t02 * pos[2]);
        inv[7]  = - (t10 * pos[0] + t11 * pos[1] + t12 * pos[2]);
        inv[11] = - (t20 * pos[0] + t21 * pos[1] + t22 * pos[2]);
        // Set the bottom row for homogeneous coordinates
        inv[12] = 0.0f; inv[13] = 0.0f; inv[14] = 0.0f; inv[15] = 1.0f;
        return inv;
    }

    EnvConstraintCuboid::EnvConstraintCuboid(
        const std::string& constraint_name,
        const std::vector<std::vector<float>>& env_collision_cuboid_pos,
        const std::vector<std::vector<float>>& env_collision_cuboid_orientation,
        const std::vector<std::vector<float>>& env_collision_cuboid_max,
        const std::vector<std::vector<float>>& env_collision_cuboid_min
    )
    : BaseConstraint(constraint_name, false) // This constraint is not projectable.
    {
        num_of_env_collision_cuboids = env_collision_cuboid_pos.size();

        // Precompute the inverse pose matrices for each cuboid.
        // Each cuboid's inverse pose is a 4x4 matrix (16 floats).
        std::vector<float> inv_pose_matrices;
        inv_pose_matrices.reserve(num_of_env_collision_cuboids * 16);
        for (int i = 0; i < num_of_env_collision_cuboids; i++) {
            // Compute the inverse pose matrix from the cuboid's position and orientation.
            std::vector<float> inv = computeInversePoseMatrix(env_collision_cuboid_pos[i], env_collision_cuboid_orientation[i]);
            inv_pose_matrices.insert(inv_pose_matrices.end(), inv.begin(), inv.end());
        }

        // Allocate device memory for inverse pose matrices and cuboid extents.
        size_t inv_pose_bytes = static_cast<size_t>(num_of_env_collision_cuboids) * sizeof(float) * 16;
        size_t extents_bytes = static_cast<size_t>(num_of_env_collision_cuboids) * sizeof(float) * 3;
        cudaMalloc(&d_env_collision_cuboids_inverse_pose_matrix_in_base_link, inv_pose_bytes);
        cudaMalloc(&d_env_collision_cuboids_max, extents_bytes);
        cudaMalloc(&d_env_collision_cuboids_min, extents_bytes);

        // Copy the computed inverse pose matrices to the device.
        cudaMemcpy(d_env_collision_cuboids_inverse_pose_matrix_in_base_link, inv_pose_matrices.data(), inv_pose_bytes, cudaMemcpyHostToDevice);

        // Flatten the max and min extents and copy them to the device.
        std::vector<float> flattened_max = floatVectorFlatten(env_collision_cuboid_max);
        std::vector<float> flattened_min = floatVectorFlatten(env_collision_cuboid_min);
        cudaMemcpy(d_env_collision_cuboids_max, flattened_max.data(), extents_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_env_collision_cuboids_min, flattened_min.data(), extents_bytes, cudaMemcpyHostToDevice);
    }

    EnvConstraintCuboid::~EnvConstraintCuboid()
    {
        cudaFree(d_env_collision_cuboids_inverse_pose_matrix_in_base_link);
        cudaFree(d_env_collision_cuboids_max);
        cudaFree(d_env_collision_cuboids_min);
    }

    // CUDA kernel to compute the collision cost between self-collision spheres and environment cuboids.
    __global__ void computeAndSumCuboidCollisionCostKernel(
        const float* __restrict__ d_self_collision_spheres_pos_in_base_link, // [num_configurations x num_self_collision_spheres x 3]
        const float* __restrict__ d_self_collision_spheres_radius,            // [num_self_collision_spheres]
        int num_of_self_collision_spheres,
        int num_of_configurations,
        const float* __restrict__ d_env_collision_cuboids_inverse_pose_matrix, // [num_cuboids x 16]
        const float* __restrict__ d_env_collision_cuboids_min,                 // [num_cuboids x 3]
        const float* __restrict__ d_env_collision_cuboids_max,                 // [num_cuboids x 3]
        int num_of_env_collision_cuboids,
        float* d_cost                                                          // [num_configurations]
    )
    {
        extern __shared__ float sdata[];

        int state_index = blockIdx.x;
        if (state_index >= num_of_configurations) return;

        int tid = threadIdx.x;
        int totalPairs = num_of_self_collision_spheres * num_of_env_collision_cuboids;
        float localSum = 0.0f;

        for (int pairIdx = tid; pairIdx < totalPairs; pairIdx += blockDim.x) {
            int self_idx = pairIdx / num_of_env_collision_cuboids;
            int cuboid_idx = pairIdx % num_of_env_collision_cuboids;

            int pos_index = state_index * num_of_self_collision_spheres * 3 + self_idx * 3;
            float sphere_x = d_self_collision_spheres_pos_in_base_link[pos_index + 0];
            float sphere_y = d_self_collision_spheres_pos_in_base_link[pos_index + 1];
            float sphere_z = d_self_collision_spheres_pos_in_base_link[pos_index + 2];
            float sphere_radius = d_self_collision_spheres_radius[self_idx];

            int mat_index = cuboid_idx * 16;
            float m0 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 0];
            float m1 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 1];
            float m2 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 2];
            float m3 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 3];
            float m4 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 4];
            float m5 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 5];
            float m6 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 6];
            float m7 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 7];
            float m8 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 8];
            float m9 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 9];
            float m10 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 10];
            float m11 = d_env_collision_cuboids_inverse_pose_matrix[mat_index + 11];

            float local_x = m0 * sphere_x + m1 * sphere_y + m2 * sphere_z + m3;
            float local_y = m4 * sphere_x + m5 * sphere_y + m6 * sphere_z + m7;
            float local_z = m8 * sphere_x + m9 * sphere_y + m10 * sphere_z + m11;

            int ext_index = cuboid_idx * 3;
            float cuboid_min_x = d_env_collision_cuboids_min[ext_index + 0];
            float cuboid_min_y = d_env_collision_cuboids_min[ext_index + 1];
            float cuboid_min_z = d_env_collision_cuboids_min[ext_index + 2];
            float cuboid_max_x = d_env_collision_cuboids_max[ext_index + 0];
            float cuboid_max_y = d_env_collision_cuboids_max[ext_index + 1];
            float cuboid_max_z = d_env_collision_cuboids_max[ext_index + 2];

            // Step 1: Compute the closest point on the cuboid's AABB
            float closest_x = fminf(fmaxf(local_x, cuboid_min_x), cuboid_max_x);
            float closest_y = fminf(fmaxf(local_y, cuboid_min_y), cuboid_max_y);
            float closest_z = fminf(fmaxf(local_z, cuboid_min_z), cuboid_max_z);

            // Step 2: Compute distance to closest point
            float dx = local_x - closest_x;
            float dy = local_y - closest_y;
            float dz = local_z - closest_z;
            float dist_squared = dx * dx + dy * dy + dz * dz;

            float cost = 0.0f;
            if (dist_squared < sphere_radius * sphere_radius) {
                float dist = sqrtf(dist_squared);
                cost = sphere_radius - dist;
            }

            localSum += cost;
        }

        sdata[tid] = localSum;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0)
            d_cost[state_index] = sdata[0];
    }

    void EnvConstraintCuboid::computeCost(BaseStatesPtr states)
    {
        // Cast the state and space info to the specific types.
        SingleArmSpaceInfoPtr space_info = std::static_pointer_cast<SingleArmSpaceInfo>(states->getSpaceInfo());
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

        // Retrieve the index of the constraint in the cost vector.
        int constraint_index = getConstraintIndex(space_info);
        if (constraint_index == -1)
        {
            printf("Constraint %s is not found in the space\n", this->constraint_name.c_str());
            return;
        }

        // Get the pointer to the cost buffer for this constraint.
        float* d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);

        // Setup kernel launch parameters.
        int threadsPerBlock = 256;
        int blocksPerGrid = single_arm_states->getNumOfStates(); // one block per configuration
        size_t sharedMemSize = threadsPerBlock * sizeof(float);

        // Launch the CUDA kernel.
        computeAndSumCuboidCollisionCostKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(),
            space_info->d_self_collision_spheres_radius,
            space_info->num_of_self_collision_spheres,
            single_arm_states->getNumOfStates(),
            d_env_collision_cuboids_inverse_pose_matrix_in_base_link,
            d_env_collision_cuboids_min,
            d_env_collision_cuboids_max,
            num_of_env_collision_cuboids,
            d_cost_of_current_constraint
        );
    }

} // namespace CUDAMPLib

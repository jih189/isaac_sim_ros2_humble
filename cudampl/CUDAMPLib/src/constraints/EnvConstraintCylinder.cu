#include <constraints/EnvConstraintCylinder.h>
#include <chrono>
#include <cmath>
#include <vector>

/**
 * @brief Computes a 3×4 inverse pose matrix (in row-major order) given a position and a rotation matrix.
 * 
 * The inverse is computed as:
 *   - The rotation part is the transpose of the provided 3×3 rotation matrix.
 *   - The translation part is -Rᵀ * pos.
 *
 * @param pos A vector of 3 floats representing the position.
 * @param orientation A vector of 9 floats representing a 3×3 rotation matrix in row-major order.
 * @return std::vector<float> A 12-element vector representing the 3×4 inverse pose matrix.
 */
std::vector<float> computeInversePoseMatrix(const std::vector<float>& pos, const std::vector<float>& orientation) {
    // Allocate 12 elements for the 3x4 matrix.
    std::vector<float> inv(12, 0.0f);
    
    // Extract the rotation matrix R from orientation (row-major).
    float r00 = orientation[0], r01 = orientation[1], r02 = orientation[2];
    float r10 = orientation[3], r11 = orientation[4], r12 = orientation[5];
    float r20 = orientation[6], r21 = orientation[7], r22 = orientation[8];

    // Compute the transpose Rᵀ (which is the inverse rotation).
    float t00 = r00, t01 = r10, t02 = r20;
    float t10 = r01, t11 = r11, t12 = r21;
    float t20 = r02, t21 = r12, t22 = r22;

    // Compute the translation component: -Rᵀ * pos.
    float trans0 = -(t00 * pos[0] + t01 * pos[1] + t02 * pos[2]);
    float trans1 = -(t10 * pos[0] + t11 * pos[1] + t12 * pos[2]);
    float trans2 = -(t20 * pos[0] + t21 * pos[1] + t22 * pos[2]);

    // Fill the 3x4 matrix in row-major order.
    // Row 0: [t00, t01, t02, trans0]
    inv[0] = t00;  inv[1] = t01;  inv[2]  = t02;  inv[3]  = trans0;
    // Row 1: [t10, t11, t12, trans1]
    inv[4] = t10;  inv[5] = t11;  inv[6]  = t12;  inv[7]  = trans1;
    // Row 2: [t20, t21, t22, trans2]
    inv[8] = t20;  inv[9] = t21;  inv[10] = t22;  inv[11] = trans2;

    return inv;
}

namespace CUDAMPLib {

    EnvConstraintCylinder::EnvConstraintCylinder(
        const std::string& constraint_name,
        const std::vector<std::vector<float>>& env_collision_cylinders_pos,
        const std::vector<std::vector<float>>& env_collision_cylinders_orientation,
        const std::vector<float>& env_collision_cylinders_radius,
        const std::vector<float>& env_collision_cylinders_height
    )
    : BaseConstraint(constraint_name, false) // This constraint is not projectable.
    {
        num_of_env_collision_cylinders = env_collision_cylinders_pos.size();

        // Prepare host memory for the inverse pose matrices.
        // Each cylinder gets a 3×4 matrix (12 floats).
        std::vector<float> h_inverse_pose_matrices(num_of_env_collision_cylinders * 12, 0.0f);
        for (int i = 0; i < num_of_env_collision_cylinders; i++) {
            std::vector<float> invMat = computeInversePoseMatrix(env_collision_cylinders_pos[i], env_collision_cylinders_orientation[i]); // Returns 12 elements.
            for (int j = 0; j < 12; j++) {
                h_inverse_pose_matrices[i * 12 + j] = invMat[j];
            }
        }

        // Allocate device memory and copy the inverse pose matrices.
        size_t inverse_pose_bytes = num_of_env_collision_cylinders * 12 * sizeof(float);
        cudaMalloc(&d_env_collision_cylinders_inverse_pose_matrix_in_base_link, inverse_pose_bytes);
        cudaMemcpy(d_env_collision_cylinders_inverse_pose_matrix_in_base_link,
                   h_inverse_pose_matrices.data(), inverse_pose_bytes, cudaMemcpyHostToDevice);

        // Allocate and copy cylinder radii.
        size_t radii_bytes = num_of_env_collision_cylinders * sizeof(float);
        cudaMalloc(&d_env_collision_cylinders_radius, radii_bytes);
        cudaMemcpy(d_env_collision_cylinders_radius, env_collision_cylinders_radius.data(),
                   radii_bytes, cudaMemcpyHostToDevice);

        // Allocate and copy cylinder heights.
        size_t heights_bytes = num_of_env_collision_cylinders * sizeof(float);
        cudaMalloc(&d_env_collision_cylinders_height, heights_bytes);
        cudaMemcpy(d_env_collision_cylinders_height, env_collision_cylinders_height.data(),
                   heights_bytes, cudaMemcpyHostToDevice);
    }

    EnvConstraintCylinder::~EnvConstraintCylinder() {
        cudaFree(d_env_collision_cylinders_inverse_pose_matrix_in_base_link);
        cudaFree(d_env_collision_cylinders_radius);
        cudaFree(d_env_collision_cylinders_height);
    }

    /**
     * @brief CUDA kernel to compute and sum collision cost between self-collision spheres and cylinders.
     *
     * For each self-collision sphere and each environment cylinder:
     *   1. Transform the sphere center (in base_link coordinates) into the cylinder's local frame.
     *   2. Compute the signed distance function (sdf) for a finite cylinder (aligned along the local z-axis).
     *      - Let d_xy = sqrt(x² + y²) and half_height = cylinder_height / 2.
     *      - dx = d_xy - cylinder_radius, dz = |z| - half_height.
     *      - sdf = sqrt(max(dx, 0)² + max(dz, 0)²) + min(max(dx, dz), 0).
     *   3. The penetration is defined as: penetration = sphere_radius - sdf.
     *      If penetration > 0, it is added to the cost.
     */
    __global__ void computeAndSumCylinderCollisionCostKernel(
        const float* __restrict__ d_self_collision_spheres_pos_in_base_link, // [num_configurations x num_self_collision_spheres x 3]
        const float* __restrict__ d_self_collision_spheres_radius,            // [num_self_collision_spheres]
        int num_of_self_collision_spheres,
        int num_of_configurations,
        const float* __restrict__ d_env_inverse_pose_matrices, // [num_of_cylinders x 12]
        const float* __restrict__ d_env_collision_cylinders_radius, // [num_of_cylinders]
        const float* __restrict__ d_env_collision_cylinders_height,   // [num_of_cylinders]
        int num_of_env_collision_cylinders,
        float* d_cost                                                         // [num_configurations]
    ) {
        extern __shared__ float sdata[];

        // Each block processes one configuration.
        int state_index = blockIdx.x;
        if (state_index >= num_of_configurations)
            return;

        int tid = threadIdx.x;
        int totalPairs = num_of_self_collision_spheres * num_of_env_collision_cylinders;
        float localSum = 0.0f;

        // Process all (sphere, cylinder) pairs in a strided loop.
        for (int pairIdx = tid; pairIdx < totalPairs; pairIdx += blockDim.x) {
            int sphere_idx = pairIdx / num_of_env_collision_cylinders;
            int cyl_idx = pairIdx % num_of_env_collision_cylinders;

            // Load sphere center and radius.
            int pos_index = state_index * num_of_self_collision_spheres * 3 + sphere_idx * 3;
            float p_x = d_self_collision_spheres_pos_in_base_link[pos_index + 0];
            float p_y = d_self_collision_spheres_pos_in_base_link[pos_index + 1];
            float p_z = d_self_collision_spheres_pos_in_base_link[pos_index + 2];
            float sphere_radius = d_self_collision_spheres_radius[sphere_idx];

            // Load the inverse pose matrix for the cylinder (3×4).
            int matrix_index = cyl_idx * 12;
            float m0  = d_env_inverse_pose_matrices[matrix_index + 0];
            float m1  = d_env_inverse_pose_matrices[matrix_index + 1];
            float m2  = d_env_inverse_pose_matrices[matrix_index + 2];
            float m3  = d_env_inverse_pose_matrices[matrix_index + 3];
            float m4  = d_env_inverse_pose_matrices[matrix_index + 4];
            float m5  = d_env_inverse_pose_matrices[matrix_index + 5];
            float m6  = d_env_inverse_pose_matrices[matrix_index + 6];
            float m7  = d_env_inverse_pose_matrices[matrix_index + 7];
            float m8  = d_env_inverse_pose_matrices[matrix_index + 8];
            float m9  = d_env_inverse_pose_matrices[matrix_index + 9];
            float m10 = d_env_inverse_pose_matrices[matrix_index + 10];
            float m11 = d_env_inverse_pose_matrices[matrix_index + 11];

            // Transform the sphere center into the cylinder's local frame.
            float local_x = m0 * p_x + m1 * p_y + m2 * p_z + m3;
            float local_y = m4 * p_x + m5 * p_y + m6 * p_z + m7;
            float local_z = m8 * p_x + m9 * p_y + m10 * p_z + m11;

            // Load cylinder parameters.
            float cyl_radius = d_env_collision_cylinders_radius[cyl_idx];
            float cyl_height = d_env_collision_cylinders_height[cyl_idx];
            float half_height = cyl_height * 0.5f;

            // Compute the signed distance function (sdf) for a finite cylinder.
            float d_xy = sqrtf(local_x * local_x + local_y * local_y);
            float dx = d_xy - cyl_radius;
            float dz = fabsf(local_z) - half_height;
            float outside_x = dx > 0.0f ? dx : 0.0f;
            float outside_z = dz > 0.0f ? dz : 0.0f;
            float outside = sqrtf(outside_x * outside_x + outside_z * outside_z);
            float inside = fminf(fmaxf(dx, dz), 0.0f);
            float sdf = outside + inside;

            // Compute penetration and cost.
            float penetration = sphere_radius - sdf;
            float cost = penetration > 0.0f ? penetration : 0.0f;
            localSum += cost;
        }

        // Perform reduction in shared memory.
        sdata[tid] = localSum;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write the accumulated cost for this configuration.
        if (tid == 0) {
            d_cost[state_index] = sdata[0];
        }
    }

    void EnvConstraintCylinder::computeCost(BaseStatesPtr states) {
        // Cast the state and space info to specific types.
        SingleArmSpaceInfoPtr space_info = std::static_pointer_cast<SingleArmSpaceInfo>(states->getSpaceInfo());
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

        // Retrieve the cost buffer for this constraint.
        int constraint_index = getConstraintIndex(space_info);
        if (constraint_index == -1) {
            printf("Constraint %s is not found in the space\n", this->constraint_name.c_str());
            return;
        }
        float* d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);

        // Configure CUDA kernel launch parameters.
        int threadsPerBlock = 256;
        int blocksPerGrid = single_arm_states->getNumOfStates();  // One block per configuration.
        size_t sharedMemSize = threadsPerBlock * sizeof(float);

        // Launch the kernel.
        computeAndSumCylinderCollisionCostKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            single_arm_states->getSelfCollisionSpheresPosInBaseLinkCuda(),
            space_info->d_self_collision_spheres_radius,
            space_info->num_of_self_collision_spheres,
            single_arm_states->getNumOfStates(),
            d_env_collision_cylinders_inverse_pose_matrix_in_base_link,
            d_env_collision_cylinders_radius,
            d_env_collision_cylinders_height,
            num_of_env_collision_cylinders,
            d_cost_of_current_constraint
        );
    }

    std::string EnvConstraintCylinder::generateCheckConstraintCode()
    {
        return "// EnvConstraintCylinder check function\n";
    }

    std::string EnvConstraintCylinder::generateLaunchCheckConstraintCode()
    {
        return "// Launch EnvConstraintCylinder check function\n";
    }
} // namespace CUDAMPLib

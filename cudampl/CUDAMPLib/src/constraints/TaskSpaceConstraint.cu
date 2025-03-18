#include <constraints/TaskSpaceConstraint.h>

#include <chrono>

namespace CUDAMPLib{
    TaskSpaceConstraint::TaskSpaceConstraint(
        const std::string& constraint_name,
        const int task_link_index,
        const Eigen::Matrix4d& offset_pose_in_task_link,
        const std::vector<float>& reference_frame,
        const std::vector<float>& tolerance
    ) : BaseConstraint(constraint_name, true) // This constraint is projectable.
    {
        task_link_index_ = task_link_index;
        offset_pose_in_task_link_ = offset_pose_in_task_link;
        reference_frame_ = reference_frame;
        tolerance_ = tolerance;

        size_t d_offset_pose_in_task_link_bytes = sizeof(float) * 16;
        size_t d_reference_frame_bytes = sizeof(float) * 6;
        size_t d_tolerance_bytes = sizeof(float) * 6;

        // allocate memory
        cudaMalloc(&d_offset_pose_in_task_link_, d_offset_pose_in_task_link_bytes);
        cudaMalloc(&d_reference_frame_, d_reference_frame_bytes);
        cudaMalloc(&d_tolerance_, d_tolerance_bytes);

        // copy data to device
        std::vector<float> offset_pose_in_task_link_flattened(16);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                offset_pose_in_task_link_flattened[i * 4 + j] = offset_pose_in_task_link(i, j);
            }
        }

        cudaMemcpy(d_offset_pose_in_task_link_, offset_pose_in_task_link_flattened.data(), d_offset_pose_in_task_link_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_reference_frame_, reference_frame.data(), d_reference_frame_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_tolerance_, tolerance.data(), d_tolerance_bytes, cudaMemcpyHostToDevice);
    }

    TaskSpaceConstraint::~TaskSpaceConstraint()
    {
        cudaFree(d_offset_pose_in_task_link_);
        cudaFree(d_reference_frame_);
        cudaFree(d_tolerance_);
    }

    /**
        * @brief Multiply two 4x4 matrices.
     */
    __device__ __forceinline__ void multiply4x4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
    {
        C[0] = A[0] * B[0] + A[1] * B[4] + A[2] * B[8] + A[3] * B[12];
        C[1] = A[0] * B[1] + A[1] * B[5] + A[2] * B[9] + A[3] * B[13];
        C[2] = A[0] * B[2] + A[1] * B[6] + A[2] * B[10] + A[3] * B[14];
        C[3] = A[0] * B[3] + A[1] * B[7] + A[2] * B[11] + A[3] * B[15];

        C[4] = A[4] * B[0] + A[5] * B[4] + A[6] * B[8] + A[7] * B[12];
        C[5] = A[4] * B[1] + A[5] * B[5] + A[6] * B[9] + A[7] * B[13];
        C[6] = A[4] * B[2] + A[5] * B[6] + A[6] * B[10] + A[7] * B[14];
        C[7] = A[4] * B[3] + A[5] * B[7] + A[6] * B[11] + A[7] * B[15];

        C[8] = A[8] * B[0] + A[9] * B[4] + A[10] * B[8] + A[11] * B[12];
        C[9] = A[8] * B[1] + A[9] * B[5] + A[10] * B[9] + A[11] * B[13];
        C[10] = A[8] * B[2] + A[9] * B[6] + A[10] * B[10] + A[11] * B[14];
        C[11] = A[8] * B[3] + A[9] * B[7] + A[10] * B[11] + A[11] * B[15];

        C[12] = 0.f; C[13] = 0.f; C[14] = 0.f; C[15] = 1.f;
    }

    __global__ void computeTaskSpaceCostKernel(
        const int num_of_states, // the number of states
        const float * d_link_poses_in_base_link, // [num_of_states * num_of_links * 16]
        const int num_of_links, // the number of links
        const int task_link_index, // the index of the task link
        const float * d_offset_pose_in_task_link, // [16] as a 4x4 matrix
        const float * d_reference_frame, // [6] for x, y, z, roll, pitch, yaw
        const float * d_tolerance, // [6]
        const float cost_tolerance, // if the cost is less than this value, set it to 0.0
        float * d_cost_of_current_constraint // output
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_states)
        {
            return;
        }

        // get the link pose of the task link
        float task_link_pose[16];

        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            task_link_pose[i] = d_link_poses_in_base_link[idx * num_of_links * 16 + task_link_index * 16 + i];
        }

        // compute the end-effector pose times the offset
        float offset_pose_in_task_link[16];
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            offset_pose_in_task_link[i] = d_offset_pose_in_task_link[i];
        }

        float real_task_link_pose[16];

        // multiply the end-effector pose and the offset
        multiply4x4(task_link_pose, offset_pose_in_task_link, real_task_link_pose);

        // --- Extract translation and orientation from real_task_link_pose ---
        // For a row-major 4x4 homogeneous transform, the translation is stored in the 4th column.
        float x = real_task_link_pose[3];
        float y = real_task_link_pose[7];
        float z = real_task_link_pose[11];

        // The upper-left 3x3 block represents the rotation.
        float r00 = real_task_link_pose[0];
        // float r01 = real_task_link_pose[1];
        // float r02 = real_task_link_pose[2];
        float r10 = real_task_link_pose[4];
        // float r11 = real_task_link_pose[5];
        // float r12 = real_task_link_pose[6];
        float r20 = real_task_link_pose[8];
        float r21 = real_task_link_pose[9];
        float r22 = real_task_link_pose[10];

        // Compute Euler angles (roll, pitch, yaw) using a ZYX convention.
        // (Be sure that this matches the convention used for your d_reference_frame.)
        float pitch = asinf(-r20);
        float roll  = atan2f(r21, r22);
        float yaw   = atan2f(r10, r00);

        // --- Retrieve the reference frame and tolerances ---
        float ref_x     = d_reference_frame[0];
        float ref_y     = d_reference_frame[1];
        float ref_z     = d_reference_frame[2];
        float ref_roll  = d_reference_frame[3];
        float ref_pitch = d_reference_frame[4];
        float ref_yaw   = d_reference_frame[5];

        float tol_x     = d_tolerance[0];
        float tol_y     = d_tolerance[1];
        float tol_z     = d_tolerance[2];
        float tol_roll  = d_tolerance[3];
        float tol_pitch = d_tolerance[4];
        float tol_yaw   = d_tolerance[5];

        // --- Compute normalized differences ---
        float diff_x     = x - ref_x;
        float diff_y     = y - ref_y;
        float diff_z     = z - ref_z;
        float diff_roll  = roll - ref_roll;
        float diff_pitch = pitch - ref_pitch;
        float diff_yaw   = yaw - ref_yaw;

        float cost_x = max(diff_x * diff_x - tol_x * tol_x, 0.f);
        float cost_y = max(diff_y * diff_y - tol_y * tol_y, 0.f);
        float cost_z = max(diff_z * diff_z - tol_z * tol_z, 0.f);
        float cost_roll = max(diff_roll * diff_roll - tol_roll * tol_roll, 0.f);
        float cost_pitch = max(diff_pitch * diff_pitch - tol_pitch * tol_pitch, 0.f);
        float cost_yaw = max(diff_yaw * diff_yaw - tol_yaw * tol_yaw, 0.f);

        // --- Compute Euclidean distance in 6D task space ---
        float cost = sqrtf(cost_x + cost_y + cost_z + cost_roll + cost_pitch + cost_yaw);

        if (cost < cost_tolerance)
        {
            cost = 0.0f;
        }

        // Store the computed cost in the output array.
        d_cost_of_current_constraint[idx] = cost;
    }

    void TaskSpaceConstraint::computeCost(BaseStatesPtr states)
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

        // use kernel function to compute the cost
        // each thread computes the cost of a state, and this cost is the Euclidean distance between
        // the current end-effector pose times the offset and the reference frame

        int threadsPerBlock = 256;
        int blocksPerGrid = (single_arm_states->getNumOfStates() + threadsPerBlock - 1) / threadsPerBlock;

        computeTaskSpaceCostKernel<<<blocksPerGrid, threadsPerBlock>>>(
            single_arm_states->getNumOfStates(),
            single_arm_states->getLinkPosesInBaseLinkCuda(),
            space_info->num_of_links,
            task_link_index_,
            d_offset_pose_in_task_link_,
            d_reference_frame_,
            d_tolerance_,
            CUDAMPLib_TASK_SPACE_CONSTRAINT_TOLERANCE,
            d_cost_of_current_constraint
        );

        CUDA_CHECK(cudaGetLastError()); // Check for launch errors
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    __global__ void computeGradientErrorKernel(
        const int num_of_states,               // number of states
        const float * d_link_poses_in_base_link, // [num_of_states * num_of_links * 16]
        const int num_of_links,                // number of links
        const int num_of_joint,                // total number of joints
        const int* __restrict__ joint_types,
        const float* __restrict__ joint_axes,
        const int task_link_index,             // index of the task link
        const float * d_offset_pose_in_task_link, // [16] as a 4x4 matrix
        const float * d_reference_frame,       // [6]: x, y, z, roll, pitch, yaw
        const float * d_tolerance,             // [6]
        const int * d_active_joint_map,        // [num_of_joint] binary mask: 1 means active, 0 inactive
        const int num_of_active_joints,        // number of active joints
        float * d_real_task_link_space_jacobian,     // [num_of_states * num_of_joint * 6]
        float * d_real_task_link_space_jacobian_inv, // [num_of_states * num_of_active_joints * 6]
        float * d_grad_of_current_constraint,        // gradient output (one per active joint)
        float * d_cost_of_current_constraint         // cost output
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_states) {
            return;
        }

        int config_offset = idx * num_of_links * 16;
        int jac_config_offset = idx * num_of_joint * 6;

        // --- Compute the task link pose and real task link pose --- //

        float task_link_pose[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            task_link_pose[i] = d_link_poses_in_base_link[config_offset + task_link_index * 16 + i];
        }

        float offset_pose_in_task_link[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            offset_pose_in_task_link[i] = d_offset_pose_in_task_link[i];
        }

        float real_task_link_pose[16];
        multiply4x4(task_link_pose, offset_pose_in_task_link, real_task_link_pose);

        // --- Extract translation and Euler angles (ZYX convention) --- //

        float x = real_task_link_pose[3];
        float y = real_task_link_pose[7];
        float z = real_task_link_pose[11];

        float r00 = real_task_link_pose[0];
        float r10 = real_task_link_pose[4];
        float r20 = real_task_link_pose[8];
        float r21 = real_task_link_pose[9];
        float r22 = real_task_link_pose[10];

        float pitch = asinf(-r20);
        float roll  = atan2f(r21, r22);
        float yaw   = atan2f(r10, r00);

        // --- Retrieve reference and tolerances --- //

        float ref_x     = d_reference_frame[0];
        float ref_y     = d_reference_frame[1];
        float ref_z     = d_reference_frame[2];
        float ref_roll  = d_reference_frame[3];
        float ref_pitch = d_reference_frame[4];
        float ref_yaw   = d_reference_frame[5];

        float tol_x     = d_tolerance[0];
        float tol_y     = d_tolerance[1];
        float tol_z     = d_tolerance[2];
        float tol_roll  = d_tolerance[3];
        float tol_pitch = d_tolerance[4];
        float tol_yaw   = d_tolerance[5];

        float diff_x     = ref_x - x;
        float diff_y     = ref_y - y;
        float diff_z     = ref_z - z;
        float diff_roll  = ref_roll - roll;
        float diff_pitch = ref_pitch - pitch;
        float diff_yaw   = ref_yaw - yaw;

        // --- Compute the error vector --- //
        float error_x     = copysignf(fmaxf(fabsf(diff_x)     - tol_x, 0.0f), diff_x);
        float error_y     = copysignf(fmaxf(fabsf(diff_y)     - tol_y, 0.0f), diff_y);
        float error_z     = copysignf(fmaxf(fabsf(diff_z)     - tol_z, 0.0f), diff_z);
        float error_roll  = copysignf(fmaxf(fabsf(diff_roll)  - tol_roll, 0.0f), diff_roll);
        float error_pitch = copysignf(fmaxf(fabsf(diff_pitch) - tol_pitch, 0.0f), diff_pitch);
        float error_yaw   = copysignf(fmaxf(fabsf(diff_yaw)   - tol_yaw, 0.0f), diff_yaw);

        // --- Compute the full space Jacobian (6 x num_of_joint) --- //
        for (int j = 0; j < num_of_joint; j++) {
            int jac_base_index = jac_config_offset + j * 6;
            float* J_col = &d_real_task_link_space_jacobian[jac_base_index];

            // If joint j is beyond the task link, it does not affect the end-effector.
            if (j > task_link_index) {
                #pragma unroll
                for (int r = 0; r < 6; r++) {
                    J_col[r] = 0.f;
                }
            } else {
                // Retrieve the transformation T_j for joint j.
                const float* T_j = &d_link_poses_in_base_link[config_offset + j * 16];

                // Extract the 3x3 rotation matrix from T_j.
                float R_j[9];
                R_j[0] = T_j[0];  R_j[1] = T_j[1];  R_j[2] = T_j[2];
                R_j[3] = T_j[4];  R_j[4] = T_j[5];  R_j[5] = T_j[6];
                R_j[6] = T_j[8];  R_j[7] = T_j[9];  R_j[8] = T_j[10];

                // Transform the joint axis from the local frame to the space frame.
                float axis[3] = { joint_axes[j * 3 + 0],
                                joint_axes[j * 3 + 1],
                                joint_axes[j * 3 + 2] };
                float w[3];
                w[0] = R_j[0] * axis[0] + R_j[1] * axis[1] + R_j[2] * axis[2];
                w[1] = R_j[3] * axis[0] + R_j[4] * axis[1] + R_j[5] * axis[2];
                w[2] = R_j[6] * axis[0] + R_j[7] * axis[1] + R_j[8] * axis[2];

                // Extract the position of joint j.
                float p_j[3] = { T_j[3], T_j[7], T_j[11] };

                int jt = joint_types[j];
                if (jt == CUDAMPLib_REVOLUTE) {
                    // For revolute joints: angular part = w; linear part = w x (p_i - p_j).
                    J_col[0] = w[0];
                    J_col[1] = w[1];
                    J_col[2] = w[2];
                    float d[3] = { x - p_j[0], y - p_j[1], z - p_j[2] };
                    J_col[3] = w[1] * d[2] - w[2] * d[1];
                    J_col[4] = w[2] * d[0] - w[0] * d[2];
                    J_col[5] = w[0] * d[1] - w[1] * d[0];
                }
                else if (jt == CUDAMPLib_PRISMATIC) {
                    // For prismatic joints: angular part = 0; linear part = transformed axis.
                    J_col[0] = 0.f;
                    J_col[1] = 0.f;
                    J_col[2] = 0.f;
                    J_col[3] = w[0];
                    J_col[4] = w[1];
                    J_col[5] = w[2];
                }
                else {
                    // For fixed or unknown joint types.
                    J_col[0] = J_col[1] = J_col[2] = 0.f;
                    J_col[3] = J_col[4] = J_col[5] = 0.f;
                }
            }
        }

        // --- Crop the Jacobian using the binary active joint mask --- //
        // Build A = J_active * J_active^T, where J_active is 6 x num_of_active_joints.
        // The full Jacobian is stored at offset 'jac_config_offset' in d_real_task_link_space_jacobian.
        float A[36] = {0.0f};
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                float sum = 0.0f;
                // Iterate over all joints and include only if active.
                for (int k = 0; k < num_of_joint; k++) {
                    if (d_active_joint_map[k] != 0) {
                        sum += d_real_task_link_space_jacobian[jac_config_offset + k * 6 + i] *
                            d_real_task_link_space_jacobian[jac_config_offset + k * 6 + j];
                    }
                }
                A[i * 6 + j] = sum;
            }
        }

        // --- Invert A using Gauss–Jordan elimination with damping --- //
        // Damping factor to regularize the inversion.
        const float lambda = 1e-4f;
        float A_inv[36];
        bool invertible = true;
        float aug[6][12];
        // Build augmented matrix [A_damped | I]
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                if (i == j) {
                    aug[i][j] = A[i * 6 + j] + lambda;
                }
                else {
                    aug[i][j] = A[i * 6 + j];
                }
            }
            for (int j = 6; j < 12; j++) {
                aug[i][j] = (j - 6 == i) ? 1.0f : 0.0f;
            }
        }
        // Perform Gauss–Jordan elimination.
        for (int i = 0; i < 6; i++) {
            float pivot = aug[i][i];
            if (fabsf(pivot) < 1e-8f) {
                invertible = false;
                break;
            }
            for (int j = 0; j < 12; j++) {
                aug[i][j] /= pivot;
            }
            for (int k = 0; k < 6; k++) {
                if (k != i) {
                    float factor = aug[k][i];
                    for (int j = 0; j < 12; j++) {
                        aug[k][j] -= factor * aug[i][j];
                    }
                }
            }
        }
        if (invertible) {
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    A_inv[i * 6 + j] = aug[i][j + 6];
                }
            }
        }

        // --- Compute the pseudoinverse for the cropped Jacobian --- //
        // We store the result as a (num_of_active_joints x 6) matrix per state.
        int cropped_offset = idx * num_of_active_joints * 6;
        int active_index = 0;
        for (int j = 0; j < num_of_joint; j++) {
            if (d_active_joint_map[j] != 0) {
                for (int i = 0; i < 6; i++) {
                    float sum = 0.0f;
                    for (int m = 0; m < 6; m++) {
                        sum += d_real_task_link_space_jacobian[jac_config_offset + j * 6 + m] *
                            A_inv[m * 6 + i];
                    }
                    d_real_task_link_space_jacobian_inv[cropped_offset + active_index * 6 + i] = sum;
                }
                active_index++;
            }
        }
        if (!invertible) {
            for (int i = 0; i < num_of_active_joints * 6; i++) {
                d_real_task_link_space_jacobian_inv[cropped_offset + i] = 0.0f;
            }
        }

        // --- Compute the gradient and cost --- //
        // The error vector is arranged as [roll, pitch, yaw, x, y, z].
        float error[6] = {error_roll, error_pitch, error_yaw, error_x, error_y, error_z};
        active_index = 0;
        for (int j = 0; j < num_of_joint; j++) {
            if (d_active_joint_map[j] != 0) {
                float grad_val = 0.0f;
                for (int i = 0; i < 6; i++) {
                    grad_val += d_real_task_link_space_jacobian_inv[cropped_offset + active_index * 6 + i] * error[i];
                }
                d_grad_of_current_constraint[idx * num_of_joint + j] = grad_val;
                active_index++;
            }
        }

        float cost = 0.0f;
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            cost += error[i] * error[i];
        }
        d_cost_of_current_constraint[idx] = sqrtf(cost);
    }

    void TaskSpaceConstraint::computeGradientAndError(BaseStatesPtr states)
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

        float * d_grad_of_current_constraint = &(single_arm_states->getGradientCuda()[single_arm_states->getNumOfStates() * space_info->num_of_joints * constraint_index]);
        float * d_cost_of_current_constraint = &(single_arm_states->getCostsCuda()[single_arm_states->getNumOfStates() * constraint_index]);

        // allocate memory for intermediate results
        float * d_real_task_link_space_jacobian;
        cudaMalloc(&d_real_task_link_space_jacobian, single_arm_states->getNumOfStates() * 6 * space_info->num_of_joints * sizeof(float));
        float * d_real_task_link_space_jacobian_inv;
        cudaMalloc(&d_real_task_link_space_jacobian_inv, single_arm_states->getNumOfStates() * space_info->num_of_active_joints * 6 * sizeof(float));

        // use kernel function to compute the gradient
        // each thread computes the gradient of a state
        int threadsPerBlock = 256;
        int blocksPerGrid = (single_arm_states->getNumOfStates() + threadsPerBlock - 1) / threadsPerBlock;

        computeGradientErrorKernel<<<blocksPerGrid, threadsPerBlock>>>(
            single_arm_states->getNumOfStates(),
            single_arm_states->getLinkPosesInBaseLinkCuda(),
            space_info->num_of_links,
            space_info->num_of_joints,
            space_info->d_joint_types,
            space_info->d_joint_axes,
            task_link_index_,
            d_offset_pose_in_task_link_,
            d_reference_frame_,
            d_tolerance_,
            space_info->d_active_joint_map,
            space_info->num_of_active_joints,
            d_real_task_link_space_jacobian,
            d_real_task_link_space_jacobian_inv,
            d_grad_of_current_constraint,
            d_cost_of_current_constraint
        );

        // std::vector<float> d_real_task_link_space_jacobian_host(single_arm_states->getNumOfStates() * 6 * space_info->num_of_joints);
        // cudaMemcpy(d_real_task_link_space_jacobian_host.data(), d_real_task_link_space_jacobian, single_arm_states->getNumOfStates() * 6 * space_info->num_of_joints * sizeof(float), cudaMemcpyDeviceToHost);

        // printf("task link space jacobian\n");
        // // print d_real_task_link_space_jacobian_host for debugging
        // for (int i = 0; i < single_arm_states->getNumOfStates(); i++)
        // {
        //     printf("=== %d\n", i);
        //     for (int j = 0; j < space_info->num_of_joints; j++)
        //     {
        //         printf("[%f, %f, %f, %f, %f, %f],\n", d_real_task_link_space_jacobian_host[i * 6 * space_info->num_of_joints + j * 6 + 0],
        //             d_real_task_link_space_jacobian_host[i * 6 * space_info->num_of_joints + j * 6 + 1],
        //             d_real_task_link_space_jacobian_host[i * 6 * space_info->num_of_joints + j * 6 + 2],
        //             d_real_task_link_space_jacobian_host[i * 6 * space_info->num_of_joints + j * 6 + 3],
        //             d_real_task_link_space_jacobian_host[i * 6 * space_info->num_of_joints + j * 6 + 4],
        //             d_real_task_link_space_jacobian_host[i * 6 * space_info->num_of_joints + j * 6 + 5]);
        //     }
        //     printf("\n");
        // }

        // printf("task link space jacobian inv\n");
        // std::vector<float> d_real_task_link_space_jacobian_inv_host(single_arm_states->getNumOfStates() * space_info->num_of_active_joints * 6);
        // cudaMemcpy(d_real_task_link_space_jacobian_inv_host.data(), d_real_task_link_space_jacobian_inv, single_arm_states->getNumOfStates() * space_info->num_of_active_joints * 6 * sizeof(float), cudaMemcpyDeviceToHost);

        // // print d_real_task_link_space_jacobian_inv_host for debugging
        // for (int i = 0; i < single_arm_states->getNumOfStates(); i++)
        // {
        //     printf("=== %d\n", i);
        //     for (int j = 0; j < space_info->num_of_active_joints; j++)
        //     {
        //         printf("[%f, %f, %f, %f, %f, %f],\n", d_real_task_link_space_jacobian_inv_host[i * space_info->num_of_active_joints * 6 + j * 6 + 0],
        //             d_real_task_link_space_jacobian_inv_host[i * space_info->num_of_active_joints * 6 + j * 6 + 1],
        //             d_real_task_link_space_jacobian_inv_host[i * space_info->num_of_active_joints * 6 + j * 6 + 2],
        //             d_real_task_link_space_jacobian_inv_host[i * space_info->num_of_active_joints * 6 + j * 6 + 3],
        //             d_real_task_link_space_jacobian_inv_host[i * space_info->num_of_active_joints * 6 + j * 6 + 4],
        //             d_real_task_link_space_jacobian_inv_host[i * space_info->num_of_active_joints * 6 + j * 6 + 5]);
        //     }
        //     printf("\n");
        // }

        CUDA_CHECK(cudaGetLastError()); // Check for launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        // free memory for intermediate results
        cudaFree(d_real_task_link_space_jacobian);
        cudaFree(d_real_task_link_space_jacobian_inv);

        // // convert d_grad_of_current_constraint to host
        // std::vector<float> grad_of_current_constraint(single_arm_states->getNumOfStates() * space_info->num_of_joints);
        // cudaMemcpy(grad_of_current_constraint.data(), d_grad_of_current_constraint, single_arm_states->getNumOfStates() * space_info->num_of_joints * sizeof(float), cudaMemcpyDeviceToHost);

        // // print grad_of_current_constraint
        // for (int i = 0; i < single_arm_states->getNumOfStates(); i++)
        // {
        //     printf("State %d\n", i);
        //     for (int j = 0; j < space_info->num_of_joints; j++)
        //     {
        //         printf("%f ", grad_of_current_constraint[i * space_info->num_of_joints + j]);
        //     }
        //     printf("\n");
        // }

        // // convert d_cost_of_current_constraint to host
        // std::vector<float> cost_of_current_constraint(single_arm_states->getNumOfStates());
        // cudaMemcpy(cost_of_current_constraint.data(), d_cost_of_current_constraint, single_arm_states->getNumOfStates() * sizeof(float), cudaMemcpyDeviceToHost);

        // // print cost_of_current_constraint
        // for (int i = 0; i < single_arm_states->getNumOfStates(); i++)
        // {
        //     printf("State %d: %f\n", i, cost_of_current_constraint[i]);
        // }

    }
} // namespace CUDAMPLib

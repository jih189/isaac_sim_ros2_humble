#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include <states/SingleArmStates.h>

namespace CUDAMPLib
{
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

        // Due to the fact that the last row of the transformation matrix is always [0, 0, 0, 1], we can skip the multiplication for the last row.
        C[12] = 0.f; C[13] = 0.f; C[14] = 0.f; C[15] = 1.f;
    }

    /**
        * @brief Set the matrix to identity
     */
    __device__ __forceinline__ void set_identity(float* __restrict__ M)
    {
        M[0]  = 1.f;  M[1]  = 0.f;  M[2]  = 0.f;  M[3]  = 0.f;
        M[4]  = 0.f;  M[5]  = 1.f;  M[6]  = 0.f;  M[7]  = 0.f;
        M[8]  = 0.f;  M[9]  = 0.f;  M[10] = 1.f;  M[11] = 0.f;
        M[12] = 0.f;  M[13] = 0.f;  M[14] = 0.f;  M[15] = 1.f;
    }

    /**
        * @brief Forward kinematics for a fixed joint
     */
    __device__ __forceinline__ void fixed_joint_fn_cuda(
        const float* parent_link_pose,
        const float* joint_pose,
        float* link_pose
    )
    {
        multiply4x4(parent_link_pose, joint_pose, link_pose);
    }

    __device__ __forceinline__ void make_rotation_axis_angle(
        float angle, float x, float y, float z, float* R)
    {
        // Compute the length and define a threshold.
        float length = sqrtf(x * x + y * y + z * z);
        const float thresh = 1e-12f;

        // Compute a valid flag: 1 if length is sufficient, 0 otherwise.
        // Using the ternary operator here; NVCC will typically optimize this well.
        float valid = (length >= thresh) ? 1.f : 0.f;

        // Avoid division by zero by using fmaxf. If length is too small, we use thresh.
        float inv_length = 1.f / fmaxf(length, thresh);

        // Normalize the axis. If invalid (valid==0), the result will be zero.
        float nx = x * inv_length * valid;
        float ny = y * inv_length * valid;
        float nz = z * inv_length * valid;

        // Compute trigonometric functions.
        float c = cosf(angle);
        float s = sinf(angle);
        float one_c = 1.f - c;

        // Compute the rotation matrix components from the axis–angle formula.
        // These values are valid only if valid==1; otherwise, they should be ignored.
        float r0  = c + nx * nx * one_c;
        float r1  = nx * ny * one_c - nz * s;
        float r2  = nx * nz * one_c + ny * s;
        float r4  = ny * nx * one_c + nz * s;
        float r5  = c + ny * ny * one_c;
        float r6  = ny * nz * one_c - nx * s;
        float r8  = nz * nx * one_c - ny * s;
        float r9  = nz * ny * one_c + nx * s;
        float r10 = c + nz * nz * one_c;

        // Blend the computed matrix with the identity matrix.
        // If valid==0, we output the identity matrix.
        R[0]  = r0 * valid + (1.f - valid) * 1.f;
        R[1]  = r1 * valid;
        R[2]  = r2 * valid;
        R[3]  = 0.f;
        R[4]  = r4 * valid;
        R[5]  = r5 * valid + (1.f - valid) * 1.f;
        R[6]  = r6 * valid;
        R[7]  = 0.f;
        R[8]  = r8 * valid;
        R[9]  = r9 * valid;
        R[10] = r10 * valid + (1.f - valid) * 1.f;
        R[11] = 0.f;
        R[12] = 0.f;
        R[13] = 0.f;
        R[14] = 0.f;
        R[15] = 1.f;
    }

    /**
        * @brief Generate the link pose for a revolute joint
     */
    __device__ __forceinline__ void revolute_joint_fn_cuda(
        const float* parent_link_pose,  // [16] in row-major
        const float* joint_pose,        // [16]
        const float* joint_axis,        // [3] -> (x,y,z)
        float        joint_value,       // rotation in radians
        float*       link_pose          // [16] output
    )
    {
        // 1. Build rotation matrix for the given joint angle & axis
        float joint_transform[16];
        make_rotation_axis_angle(
            joint_value, 
            joint_axis[0],
            joint_axis[1],
            joint_axis[2],
            joint_transform
        );

        // 2. Multiply: temp = parent_link_pose * joint_pose
        float temp[16];
        multiply4x4(parent_link_pose, joint_pose, temp);

        // 3. Multiply: link_pose = temp * joint_transform
        multiply4x4(temp, joint_transform, link_pose);
    }

    /**
        * @brief Generate the link pose for a prismatic joint
     */
    __device__ __forceinline__ void prism_joint_fn_cuda(
        const float* parent_link_pose,
        const float* joint_pose,
        const float* joint_axis,
        float joint_value,
        float* link_pose
    )
    {
        //------------------------------------------------------------------------------
        // 1) Compute translation matrix T(joint_axis, joint_value) in row-major order
        //------------------------------------------------------------------------------
        float x = joint_axis[0];
        float y = joint_axis[1];
        float z = joint_axis[2];

        // T is a 4x4 matrix in row-major form
        float T[16] = {
            1.0f, 0.0f, 0.0f, x * joint_value,
            0.0f, 1.0f, 0.0f, y * joint_value,
            0.0f, 0.0f, 1.0f, z * joint_value,
            0.0f, 0.0f, 0.0f, 1.0f
        };

        //------------------------------------------------------------------------------
        // 2) Multiply joint_pose * T -> call this intermediate joint_pose_T
        //------------------------------------------------------------------------------
        float joint_pose_T[16];
        multiply4x4(joint_pose, T, joint_pose_T);

        //------------------------------------------------------------------------------
        // 3) Multiply parent_link_pose * joint_pose_T -> final link_pose
        //------------------------------------------------------------------------------
        multiply4x4(parent_link_pose, joint_pose_T, link_pose);
    }

    __global__ void kin_forward_collision_spheres_kernel(
        float* joint_values, 
        int num_of_joint,
        int configuration_size,
        int* joint_types,
        float* joint_poses,
        int num_of_links,
        float* joint_axes,
        int* link_maps,
        int num_of_collision_spheres,
        int* collision_spheres_map,
        float* collision_spheres_pos,
        float* link_poses_set,
        float* collision_spheres_pos_in_baselink
    ) 
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < configuration_size) {

            // set the first link pose to identity matrix because it is the base link
            set_identity(&link_poses_set[idx * num_of_links * 16]);

            // Calculate forward kinematics for each link
            for (size_t i = 1; i < num_of_links; i++) // The first link is the base link, so we can skip it
            {
                float* parent_link_pose = &link_poses_set[idx * num_of_links * 16 + link_maps[i] * 16];
                float* current_link_pose = &link_poses_set[idx * num_of_links * 16 + i * 16];
                // based on the joint type, calculate the link pose
                switch (joint_types[i])
                {
                    case CUDAMPLib_REVOLUTE:
                        revolute_joint_fn_cuda(parent_link_pose, &joint_poses[i * 16], &joint_axes[i * 3], joint_values[idx * num_of_joint + i], current_link_pose);
                        // j++;
                        break;
                    case CUDAMPLib_PRISMATIC:
                        prism_joint_fn_cuda(parent_link_pose, &joint_poses[i * 16], &joint_axes[i * 3], joint_values[idx * num_of_joint + i], current_link_pose);
                        // j++;
                        break;
                    case CUDAMPLib_FIXED:
                        fixed_joint_fn_cuda(parent_link_pose, &joint_poses[i * 16], current_link_pose);
                        break;
                    default:
                        printf("Unknown joint type: %d\n", joint_types[i]);
                        break;
                }
            }

            // Calculate the collision spheres position in the base link frame
            for (size_t i = 0; i < num_of_collision_spheres; i++)
            {
                float* collision_sphere_pos = &collision_spheres_pos[i * 3]; // collision sphere position in link frame
                float* collision_sphere_pos_in_baselink = &collision_spheres_pos_in_baselink[idx * num_of_collision_spheres * 3 + i * 3]; // collision sphere position in base link frame
                float* link_pose = &link_poses_set[idx * num_of_links * 16 + collision_spheres_map[i] * 16]; // link pose in base link frame

                collision_sphere_pos_in_baselink[0] = link_pose[0] * collision_sphere_pos[0] + link_pose[1] * collision_sphere_pos[1] + link_pose[2] * collision_sphere_pos[2] + link_pose[3];
                collision_sphere_pos_in_baselink[1] = link_pose[4] * collision_sphere_pos[0] + link_pose[5] * collision_sphere_pos[1] + link_pose[6] * collision_sphere_pos[2] + link_pose[7];
                collision_sphere_pos_in_baselink[2] = link_pose[8] * collision_sphere_pos[0] + link_pose[9] * collision_sphere_pos[1] + link_pose[10] * collision_sphere_pos[2] + link_pose[11];
            }
        }
    }

    __global__ void kin_forward_kernel(
        const float* __restrict__ joint_values, 
        const int num_of_joint,
        const int configuration_size,
        const int* __restrict__ joint_types,
        const float* __restrict__ joint_poses,
        const int num_of_links,
        const float* __restrict__ joint_axes,
        const int* __restrict__ link_maps,
        float* __restrict__ link_poses_set
    ) 
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= configuration_size) {
            return;
        }

        // set the first link pose to identity matrix because it is the base link
        set_identity(&link_poses_set[idx * num_of_links * 16]);

        // Calculate forward kinematics for each link
        for (size_t i = 1; i < num_of_links; i++) // The first link is the base link, so we can skip it
        {
            float* parent_link_pose = &link_poses_set[idx * num_of_links * 16 + link_maps[i] * 16];
            float* current_link_pose = &link_poses_set[idx * num_of_links * 16 + i * 16];
            // based on the joint type, calculate the link pose
            int j_type = joint_types[i];
            switch (j_type)
            {
                case CUDAMPLib_REVOLUTE:
                    revolute_joint_fn_cuda(parent_link_pose, &joint_poses[i * 16], &joint_axes[i * 3], joint_values[idx * num_of_joint + i], current_link_pose);
                    break;
                case CUDAMPLib_PRISMATIC:
                    prism_joint_fn_cuda(parent_link_pose, &joint_poses[i * 16], &joint_axes[i * 3], joint_values[idx * num_of_joint + i], current_link_pose);
                    break;
                case CUDAMPLib_FIXED:
                    fixed_joint_fn_cuda(parent_link_pose, &joint_poses[i * 16], current_link_pose);
                    break;
                // default: // This will waste compution resource.
                //     printf("Unknown joint type: %d\n", joint_types[i]);
                //     break;
            }
        }
    }

    __global__ void kin_space_jacobian_per_link_kernel(
        const int configuration_size,
        const int num_of_joint,
        const int num_of_links,
        const int* __restrict__ joint_types,
        const float* __restrict__ joint_axes,
        const float* __restrict__ link_poses_set,
        float* __restrict__ space_jacobians // [configuration][link][joint][6]
    )
    {
        // Each thread processes one link of one configuration.
        int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
        int total_links = configuration_size * num_of_links;
        if (global_idx >= total_links) {
            return;
        }
        
        // Determine which configuration and which link.
        int config = global_idx / num_of_links;
        int i = global_idx % num_of_links;
        
        // Offsets for the configuration data.
        int config_offset = config * num_of_links * 16;         // 16 floats per link pose.
        int jac_config_offset = config * num_of_links * 6 * num_of_joint; // 6 floats per joint per link.
        
        // For the base link (index 0), the Jacobian is zero.
        if (i == 0) {
            for (int j = 0; j < num_of_joint; j++) {
                int base_jac_offset = jac_config_offset + (i * 6 * num_of_joint) + (j * 6);
                #pragma unroll
                for (int r = 0; r < 6; r++) {
                    space_jacobians[base_jac_offset + r] = 0.f;
                }
            }
            return;
        }
        
        // Get the current link pose and extract its position.
        const float* current_link_pose = &link_poses_set[config_offset + i * 16];
        float p_i[3] = { current_link_pose[3], current_link_pose[7], current_link_pose[11] };
        
        // Compute the Jacobian for link i for every joint j.
        for (int j = 0; j < num_of_joint; j++) {
            int jac_base_index = jac_config_offset + (i * 6 * num_of_joint) + (j * 6);
            
            // If the joint index is greater than the link index, it does not affect the link.
            if (j > i) {
                #pragma unroll
                for (int r = 0; r < 6; r++) {
                    space_jacobians[jac_base_index + r] = 0.f;
                }
                continue;
            }
            
            // Joint 0 (assumed fixed base) contributes nothing.
            if (j == 0) {
                #pragma unroll
                for (int r = 0; r < 6; r++) {
                    space_jacobians[jac_base_index + r] = 0.f;
                }
                continue;
            }
            
            // Retrieve the transformation for joint j.
            const float* T_j = &link_poses_set[config_offset + j * 16];
            
            // Extract the 3x3 rotation matrix from T_j (row-major order).
            float R_j[9];
            R_j[0] = T_j[0];  R_j[1] = T_j[1];  R_j[2] = T_j[2];
            R_j[3] = T_j[4];  R_j[4] = T_j[5];  R_j[5] = T_j[6];
            R_j[6] = T_j[8];  R_j[7] = T_j[9];  R_j[8] = T_j[10];
            
            // Transform the joint axis into the space frame.
            float axis[3] = { joint_axes[j * 3 + 0], joint_axes[j * 3 + 1], joint_axes[j * 3 + 2] };
            float w[3];
            w[0] = R_j[0] * axis[0] + R_j[1] * axis[1] + R_j[2] * axis[2];
            w[1] = R_j[3] * axis[0] + R_j[4] * axis[1] + R_j[5] * axis[2];
            w[2] = R_j[6] * axis[0] + R_j[7] * axis[1] + R_j[8] * axis[2];
            
            // Extract the position of joint j.
            float p_j[3] = { T_j[3], T_j[7], T_j[11] };
            
            float J_col[6];
            int jt = joint_types[j];
            
            if (jt == CUDAMPLib_REVOLUTE) {
                // For revolute joints: angular part is w, linear part is w x (p_i - p_j).
                J_col[0] = w[0];
                J_col[1] = w[1];
                J_col[2] = w[2];
                float d[3] = { p_i[0] - p_j[0], p_i[1] - p_j[1], p_i[2] - p_j[2] };
                J_col[3] = w[1] * d[2] - w[2] * d[1];
                J_col[4] = w[2] * d[0] - w[0] * d[2];
                J_col[5] = w[0] * d[1] - w[1] * d[0];
            }
            else if (jt == CUDAMPLib_PRISMATIC) {
                // For prismatic joints: angular part is zero, linear part is the transformed axis.
                J_col[0] = 0.f;
                J_col[1] = 0.f;
                J_col[2] = 0.f;
                J_col[3] = w[0];
                J_col[4] = w[1];
                J_col[5] = w[2];
            }
            else {
                // For fixed or unknown joint types, the column is zero.
                J_col[0] = 0.f;  J_col[1] = 0.f;  J_col[2] = 0.f;
                J_col[3] = 0.f;  J_col[4] = 0.f;  J_col[5] = 0.f;
            }
            
            // Write the computed Jacobian column into the global array.
            #pragma unroll
            for (int r = 0; r < 6; r++) {
                space_jacobians[jac_base_index + r] = J_col[r];
            }
        }
    }

    __global__ void update_collision_spheres_kernel(
        const int num_of_states,
        const int num_of_links,
        const int num_of_self_collision_spheres,
        const int* __restrict__ collision_spheres_map,
        const float* __restrict__ collision_spheres_pos, // collision sphere position in link frame
        const float* __restrict__ link_poses_set,
        float* __restrict__ collision_spheres_pos_in_baselink
    )
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_of_states * num_of_self_collision_spheres)
            return;

        // Compute state and sphere indices
        int state_idx = idx / num_of_self_collision_spheres;
        int sphere_idx = idx % num_of_self_collision_spheres;

        // Load collision sphere position into registers
        int posIndex = sphere_idx * 3;
        float cs_x = collision_spheres_pos[posIndex];
        float cs_y = collision_spheres_pos[posIndex + 1];
        float cs_z = collision_spheres_pos[posIndex + 2];

        // Compute output index
        int outIndex = state_idx * num_of_self_collision_spheres * 3 + sphere_idx * 3;

        // Get the link index for this sphere and compute the starting index of its transformation matrix
        int link_idx = collision_spheres_map[sphere_idx];
        int linkPoseIndex = state_idx * num_of_links * 16 + link_idx * 16;

        // Load the transformation matrix elements into registers
        float m0  = link_poses_set[linkPoseIndex];       // matrix row 0, col 0
        float m1  = link_poses_set[linkPoseIndex + 1];     // matrix row 0, col 1
        float m2  = link_poses_set[linkPoseIndex + 2];     // matrix row 0, col 2
        float m3  = link_poses_set[linkPoseIndex + 3];     // matrix row 0, col 3
        float out_x = m0 * cs_x + m1 * cs_y + m2 * cs_z + m3;

        float m4  = link_poses_set[linkPoseIndex + 4];     // matrix row 1, col 0
        float m5  = link_poses_set[linkPoseIndex + 5];     // matrix row 1, col 1
        float m6  = link_poses_set[linkPoseIndex + 6];     // matrix row 1, col 2
        float m7  = link_poses_set[linkPoseIndex + 7];     // matrix row 1, col 3
        float out_y = m4 * cs_x + m5 * cs_y + m6 * cs_z + m7;

        float m8  = link_poses_set[linkPoseIndex + 8];     // matrix row 2, col 0
        float m9  = link_poses_set[linkPoseIndex + 9];     // matrix row 2, col 1
        float m10 = link_poses_set[linkPoseIndex + 10];    // matrix row 2, col 2
        float m11 = link_poses_set[linkPoseIndex + 11];    // matrix row 2, col 3
        float out_z = m8 * cs_x + m9 * cs_y + m10 * cs_z + m11;

        // Write the results to global memory
        collision_spheres_pos_in_baselink[outIndex]     = out_x;
        collision_spheres_pos_in_baselink[outIndex + 1] = out_y;
        collision_spheres_pos_in_baselink[outIndex + 2] = out_z;
    }
        

    // kernel to calculate the distance between two states
    __global__ void calculate_joint_state_distance(
        float * d_states_1, int num_of_states_1,
        float * d_states_2, int num_of_states_2, 
        int num_of_joints, int * d_active_joint_map, float * d_distances) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_of_states_1 * num_of_states_2)
            return;

        int state_1_idx = idx / num_of_states_2;
        int state_2_idx = idx % num_of_states_2;

        float sum = 0.0f;

        for (int i = 0; i < num_of_joints; i++) {
            // if (d_active_joint_map[i] != 0) {
            float diff = d_states_1[state_1_idx * num_of_joints + i] - d_states_2[state_2_idx * num_of_joints + i];
            sum += diff * diff;
            // }
        }

        d_distances[idx] = sqrtf(sum);
    }
    
    SingleArmStates::SingleArmStates(int num_of_states, SingleArmSpaceInfoPtr space_info)
    : BaseStates(num_of_states, space_info)
    {
        // check if base class is initialized
        if (isValid() == false)
        {
            // If not initialized, return.
            std::cerr << "BaseStates is not initialized" << std::endl;
            return;
        }

        this->num_of_joints = space_info->num_of_joints;

        // Allocate memory for the joint states
        size_t d_joint_states_bytes = (size_t)num_of_states * this->num_of_joints * sizeof(float);
        size_t d_link_poses_in_base_link_bytes = (size_t)num_of_states * space_info->num_of_links * 4 * 4 * sizeof(float);
        size_t d_space_jacobian_in_base_link_bytes = (size_t)num_of_states * space_info->num_of_links * 6 * space_info->num_of_joints * sizeof(float);
        size_t d_self_collision_spheres_pos_in_base_link_bytes = (size_t)num_of_states * space_info->num_of_self_collision_spheres * 3 * sizeof(float);
        size_t d_gradient_bytes = (size_t)num_of_states * this->num_of_joints * space_info->num_of_constraints * sizeof(float);
        size_t d_total_gradient_bytes = (size_t)num_of_states * this->num_of_joints * sizeof(float);

        auto allocate_result = cudaMalloc(&d_joint_states, d_joint_states_bytes);
        if (allocate_result != cudaSuccess) {
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            std::cerr << "Free memory: " << free_memory / (1024 * 1024) << " MB" << std::endl;
            std::cerr << "Requested d_joint_states_bytes: " << d_joint_states_bytes / (1024 * 1024) << " MB" << std::endl;
            // print in red
            std::cerr << "\033[31m" << "CUDA Error: " << cudaGetErrorString(allocate_result) << " d_joint_states_bytes is too large " << "\033[0m" << std::endl;

            setValid(false);
            // return;
        }
        allocate_result = cudaMalloc(&d_link_poses_in_base_link, d_link_poses_in_base_link_bytes);
        if (allocate_result != cudaSuccess) {
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            std::cerr << "Free memory: " << free_memory / (1024 * 1024) << " MB" << std::endl;
            std::cerr << "Requested d_link_poses_in_base_link_bytes: " << d_link_poses_in_base_link_bytes / (1024 * 1024) << " MB" << std::endl;
            // print in red
            std::cerr << "\033[31m" << "CUDA Error: " << cudaGetErrorString(allocate_result) << " d_link_poses_in_base_link_bytes is too large "<< "\033[0m" << std::endl;

            setValid(false);
            // return;
        }
        allocate_result = cudaMalloc(&d_self_collision_spheres_pos_in_base_link, d_self_collision_spheres_pos_in_base_link_bytes);
        if (allocate_result != cudaSuccess) {
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            std::cerr << "Free memory: " << free_memory / (1024 * 1024) << " MB" << std::endl;
            std::cerr << "Requested d_self_collision_spheres_pos_in_base_link_bytes: " << d_self_collision_spheres_pos_in_base_link_bytes / (1024 * 1024) << " MB" << std::endl;
            // print in red
            std::cerr << "\033[31m" << "CUDA Error: " << cudaGetErrorString(allocate_result) << " d_self_collision_spheres_pos_in_base_link_bytes is too large " << "\033[0m" << std::endl;

            setValid(false);
            // return;
        }
        allocate_result = cudaMalloc(&d_space_jacobian_in_base_link, d_space_jacobian_in_base_link_bytes);
        if (allocate_result != cudaSuccess) {
            cudaGetLastError();
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            std::cerr << "Free memory: " << free_memory / (1024 * 1024) << " MB" << std::endl;
            std::cerr << "Requested d_space_jacobian_in_base_link_bytes: " << d_space_jacobian_in_base_link_bytes / (1024 * 1024) << " MB" << std::endl;
            // print in red
            std::cerr << "\033[31m" << "CUDA Error: " << cudaGetErrorString(allocate_result) << " d_space_jacobian_in_base_link_bytes is too large " << "\033[0m" << std::endl;

            setValid(false);
            // return;
        }
        allocate_result = cudaMalloc(&d_gradient, d_gradient_bytes);
        if (allocate_result != cudaSuccess) {
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            std::cerr << "Free memory: " << free_memory / (1024 * 1024) << " MB" << std::endl;
            std::cerr << "Requested d_gradient_bytes: " << d_gradient_bytes / (1024 * 1024) << " MB" << std::endl;
            // print in red
            std::cerr << "\033[31m" << "CUDA Error: " << cudaGetErrorString(allocate_result) << " d_gradient_bytes is too large " << "\033[0m" << std::endl;

            setValid(false);
            // return;
        }
        allocate_result = cudaMalloc(&d_total_gradient, d_total_gradient_bytes);
        if (allocate_result != cudaSuccess) {
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            std::cerr << "Free memory: " << free_memory / (1024 * 1024) << " MB" << std::endl;
            std::cerr << "Requested d_total_gradient_bytes: " << d_total_gradient_bytes / (1024 * 1024) << " MB" << std::endl;
            // print in red
            std::cerr << "\033[31m" << "CUDA Error: " << cudaGetErrorString(allocate_result) << " d_total_gradient_bytes is too large " << "\033[0m" << std::endl;

            setValid(false);
            // return;
        }
        // CUDA_CHECK(cudaGetLastError());
    }

    SingleArmStates::~SingleArmStates()
    {
        // Free the memory
        if (num_of_states_ > 0)
        {
            cudaFree(d_joint_states);
            cudaFree(d_link_poses_in_base_link);
            cudaFree(d_space_jacobian_in_base_link);
            cudaFree(d_self_collision_spheres_pos_in_base_link);
            cudaFree(d_gradient);
            cudaFree(d_total_gradient);
            // set the pointers to nullptr for safety
            d_joint_states = nullptr;
            d_link_poses_in_base_link = nullptr;
            d_space_jacobian_in_base_link = nullptr;
            d_self_collision_spheres_pos_in_base_link = nullptr;
            d_gradient = nullptr;
            d_total_gradient = nullptr;
        }
    }

    void SingleArmStates::filterStates(const std::vector<bool> & filter_map)
    {
        int initial_num_of_states = num_of_states_;

        // call the base class filterStates
        BaseStates::filterStates(filter_map);

        size_t new_num_of_states = num_of_states_; // number of state is updated in the base class.

        if (new_num_of_states == 0){
            // Free the memory
            cudaFree(d_joint_states);
            cudaFree(d_link_poses_in_base_link);
            cudaFree(d_space_jacobian_in_base_link);
            cudaFree(d_self_collision_spheres_pos_in_base_link);
            cudaFree(d_gradient);
            cudaFree(d_total_gradient);
        }
        else{
            // static_cast the space_info to SingleArmSpaceInfo
            SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

            float * d_joint_states_new;
            float * d_link_poses_in_base_link_new;
            float * d_space_jacobian_in_base_link_new;
            float * d_self_collision_spheres_pos_in_base_link_new;
            float * d_gradient_new;
            float * d_total_gradient_new;

            size_t d_joint_states_new_bytes = new_num_of_states * this->num_of_joints * sizeof(float);
            size_t d_link_poses_in_base_link_new_bytes = new_num_of_states * single_arm_space_info->num_of_links * 4 * 4 * sizeof(float);
            size_t d_space_jacobian_in_base_link_new_bytes = new_num_of_states * single_arm_space_info->num_of_links * 6 * this->num_of_joints * sizeof(float);
            size_t d_self_collision_spheres_pos_in_base_link_new_bytes = new_num_of_states * single_arm_space_info->num_of_self_collision_spheres * 3 * sizeof(float);
            size_t d_gradient_new_bytes = new_num_of_states * this->num_of_joints * single_arm_space_info->num_of_constraints * sizeof(float);
            size_t d_total_gradient_new_bytes = new_num_of_states * this->num_of_joints * sizeof(float);

            // Allocate memory for the joint states
            cudaMalloc(&d_joint_states_new, d_joint_states_new_bytes);
            cudaMalloc(&d_link_poses_in_base_link_new, d_link_poses_in_base_link_new_bytes);
            cudaMalloc(&d_space_jacobian_in_base_link_new, d_space_jacobian_in_base_link_new_bytes);
            cudaMalloc(&d_self_collision_spheres_pos_in_base_link_new, d_self_collision_spheres_pos_in_base_link_new_bytes);
            cudaMalloc(&d_gradient_new, d_gradient_new_bytes);
            cudaMalloc(&d_total_gradient_new, d_total_gradient_new_bytes);

            // Copy the joint states from the old memory to the new memory
            int j = 0;
            for (int i = 0; i < initial_num_of_states; i++)
            {
                if (filter_map[i])
                {
                    // copy asynchrounously
                    cudaMemcpyAsync(d_joint_states_new + j * num_of_joints, d_joint_states + i * num_of_joints, num_of_joints * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(d_link_poses_in_base_link_new + j * single_arm_space_info->num_of_links * 4 * 4, d_link_poses_in_base_link + i * single_arm_space_info->num_of_links * 4 * 4, single_arm_space_info->num_of_links * 4 * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(d_space_jacobian_in_base_link_new + j * single_arm_space_info->num_of_links * 6 * num_of_joints, d_space_jacobian_in_base_link + i * single_arm_space_info->num_of_links * 6 * num_of_joints, single_arm_space_info->num_of_links * 6 * num_of_joints * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(d_self_collision_spheres_pos_in_base_link_new + j * single_arm_space_info->num_of_self_collision_spheres * 3, d_self_collision_spheres_pos_in_base_link + i * single_arm_space_info->num_of_self_collision_spheres * 3, single_arm_space_info->num_of_self_collision_spheres * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(d_gradient_new + j * num_of_joints * single_arm_space_info->num_of_constraints, d_gradient + i * num_of_joints * single_arm_space_info->num_of_constraints, num_of_joints * single_arm_space_info->num_of_constraints * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(d_total_gradient_new + j * num_of_joints, d_total_gradient + i * num_of_joints, num_of_joints * sizeof(float), cudaMemcpyDeviceToDevice);
                    j++;
                }
            }

            // Wait for the copy to finish
            cudaDeviceSynchronize();

            // Free the old memory
            cudaFree(d_joint_states);
            cudaFree(d_link_poses_in_base_link);
            cudaFree(d_space_jacobian_in_base_link);
            cudaFree(d_self_collision_spheres_pos_in_base_link);
            cudaFree(d_gradient);
            cudaFree(d_total_gradient);

            // Update the pointers
            d_joint_states = d_joint_states_new;
            d_link_poses_in_base_link = d_link_poses_in_base_link_new;
            d_space_jacobian_in_base_link = d_space_jacobian_in_base_link_new;
            d_self_collision_spheres_pos_in_base_link = d_self_collision_spheres_pos_in_base_link_new;
            d_gradient = d_gradient_new;
            d_total_gradient = d_total_gradient_new;
        }

        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<std::vector<float>> SingleArmStates::getJointStatesFullHost() const
    {
        // Allocate memory for the joint states
        std::vector<float> joint_states_flatten(num_of_states_ * num_of_joints, 0.0);

        // Copy the joint states from device to host
        cudaMemcpy(joint_states_flatten.data(), d_joint_states, num_of_states_ * num_of_joints * sizeof(float), cudaMemcpyDeviceToHost);

        // Reshape the joint states
        std::vector<std::vector<float>> joint_states(num_of_states_, std::vector<float>(num_of_joints, 0.0));
        for (int i = 0; i < num_of_states_; i++)
        {
            for (int j = 0; j < num_of_joints; j++)
            {
                joint_states[i][j] = joint_states_flatten[i * num_of_joints + j];
            }
        }

        return joint_states;
    }

    std::vector<std::vector<float>> SingleArmStates::getJointStatesHost() const
    {
        // get space info
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);
        std::vector<std::vector<float>> joint_states_full = this->getJointStatesFullHost();

        // Allocate memory for the joint states
        std::vector<std::vector<float>> joint_states;

        for (int i = 0; i < num_of_states_; i++)
        {
            std::vector<float> joint_state;
            for (int j = 0; j < num_of_joints; j++)
            {
                if (space_info_single_arm_space->active_joint_map[j] != 0)
                {
                    joint_state.push_back(joint_states_full[i][j]);
                }
            }
            joint_states.push_back(joint_state);
        }

        return joint_states;
    }

    std::vector<std::vector<std::vector<float>>> SingleArmStates::getSelfCollisionSpheresPosInBaseLinkHost()
    {
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

        // Allocate memory for the self collision spheres position in base link frame
        std::vector<float> self_collision_spheres_pos_in_base_link_flatten(num_of_states_ * space_info_single_arm_space->num_of_self_collision_spheres * 3, 0.0);

        // Copy the self collision spheres position in base link frame from device to host
        cudaMemcpy(self_collision_spheres_pos_in_base_link_flatten.data(), d_self_collision_spheres_pos_in_base_link, num_of_states_ * space_info_single_arm_space->num_of_self_collision_spheres * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // Reshape the self collision spheres position in base link frame
        std::vector<std::vector<std::vector<float>>> self_collision_spheres_pos_in_base_link(num_of_states_, std::vector<std::vector<float>>(space_info_single_arm_space->num_of_self_collision_spheres, std::vector<float>(3, 0.0)));

        for (int i = 0; i < num_of_states_; i++)
        {
            for (int j = 0; j < space_info_single_arm_space->num_of_self_collision_spheres; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    self_collision_spheres_pos_in_base_link[i][j][k] = self_collision_spheres_pos_in_base_link_flatten[i * space_info_single_arm_space->num_of_self_collision_spheres * 3 + j * 3 + k];
                }
            }
        }

        return self_collision_spheres_pos_in_base_link;
    }

    void SingleArmStates::oldUpdate()
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states_ + threadsPerBlock - 1) / threadsPerBlock;
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);
        
        // Update the states
        kin_forward_collision_spheres_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_joint_states,
            num_of_joints,
            num_of_states_,
            space_info_single_arm_space->d_joint_types,
            space_info_single_arm_space->d_joint_poses,
            space_info_single_arm_space->num_of_links,
            space_info_single_arm_space->d_joint_axes,
            space_info_single_arm_space->d_link_parent_link_maps,
            space_info_single_arm_space->num_of_self_collision_spheres,
            space_info_single_arm_space->d_collision_spheres_to_link_map,
            space_info_single_arm_space->d_self_collision_spheres_pos_in_link,
            d_link_poses_in_base_link,
            d_self_collision_spheres_pos_in_base_link
        );

        // Wait for the kernel to finish
        cudaDeviceSynchronize();
    }

    void SingleArmStates::update()
    {
        
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

        // this->calculateForwardKinematics();
        this->calculateForwardKinematicsNvrtv();

        // // compute space jacobian in base link
        // this->calculateSpaceJacobian(false);

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states_ * space_info_single_arm_space->num_of_self_collision_spheres + threadsPerBlock - 1) / threadsPerBlock;

        // update the self collision spheres position in base link frame
        update_collision_spheres_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            num_of_states_,
            space_info_single_arm_space->num_of_links,
            space_info_single_arm_space->num_of_self_collision_spheres,
            space_info_single_arm_space->d_collision_spheres_to_link_map,
            space_info_single_arm_space->d_self_collision_spheres_pos_in_link,
            d_link_poses_in_base_link,
            d_self_collision_spheres_pos_in_base_link
        );

        CUDA_CHECK(cudaGetLastError()); // Check for launch errors
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // void SingleArmStates::calculateForwardKinematicsWithSharedMemoryNvrtv()
    // {
    //     CUDA_CHECK(cudaDeviceSynchronize());
    //     CUDA_CHECK(cudaGetLastError());

    //     int threadsPerBlock = 256;
    //     int blocksPerGrid = (num_of_states_ + threadsPerBlock - 1) / threadsPerBlock;
    //     SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);
        
    //     // Set up kernel parameters.
    //     void *args[] = { &d_joint_states, &num_of_states_, &d_link_poses_in_base_link };

    //     // Launch the kernel using the member function of KernelFunction.
    //     space_info_single_arm_space->kinForwardKernelFuncPtr->launchKernel(dim3(blocksPerGrid, 1, 1),
    //                                 dim3(threadsPerBlock, 1, 1),
    //                                 threadsPerBlock * num_of_joints * sizeof(float),          // shared memory size
    //                                 nullptr,    // stream
    //                                 args);
    //     cudaDeviceSynchronize();
    //     // check for errors
    //     if (cudaGetLastError() != cudaSuccess) {
    //         // print in red
    //         std::cerr << "\033[31m" << "number of states: " << num_of_states_ << " num_of_joints: " << num_of_joints << "\033[0m" << std::endl;
    //     }
    //     CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    // }

    void SingleArmStates::calculateForwardKinematicsNvrtv()
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states_ + threadsPerBlock - 1) / threadsPerBlock;
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);
        
        // Set up kernel parameters.
        void *args[] = { &d_joint_states, &num_of_states_, &d_link_poses_in_base_link };

        // Launch the kernel using the member function of KernelFunction.
        space_info_single_arm_space->kinForwardKernelFuncPtr->launchKernel(dim3(blocksPerGrid, 1, 1),
                                    dim3(threadsPerBlock, 1, 1),
                                    0,          // shared memory size
                                    nullptr,    // stream
                                    args);
        CUDA_CHECK(cudaDeviceSynchronize()); // Check for launch errors
    }

    void SingleArmStates::calculateForwardKinematics()
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states_ + threadsPerBlock - 1) / threadsPerBlock;
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);
        
        // Update the states
        kin_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_joint_states,
            num_of_joints,
            num_of_states_,
            space_info_single_arm_space->d_joint_types,
            space_info_single_arm_space->d_joint_poses,
            space_info_single_arm_space->num_of_links,
            space_info_single_arm_space->d_joint_axes,
            space_info_single_arm_space->d_link_parent_link_maps,
            d_link_poses_in_base_link
        );

        CUDA_CHECK(cudaGetLastError()); // Check for launch errors
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void SingleArmStates::calculateSpaceJacobian(bool synchronize)
    {
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

        int threadsPerBlock = 256;
        // calculate space jacobian
        int blocksPerGrid = (num_of_states_ * space_info_single_arm_space->num_of_links + threadsPerBlock - 1) / threadsPerBlock;

        // update the space jacobian in base link frame
        kin_space_jacobian_per_link_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            num_of_states_,
            num_of_joints,
            space_info_single_arm_space->num_of_links,
            space_info_single_arm_space->d_joint_types,
            space_info_single_arm_space->d_joint_axes,
            d_link_poses_in_base_link,
            d_space_jacobian_in_base_link
        );

        if (synchronize)
        {
            CUDA_CHECK(cudaGetLastError()); // Check for launch errors
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    __global__ void sum_gradients_kernel(
        int num_of_states, 
        int num_of_joints,
        int num_of_constraints,
        int * d_constraint_indexs, // [num_of_constraint_indexs]
        int num_of_constraint_indexs,
        float * d_gradient, // [num_of_states * num_of_joints * num_of_constraints]
        float * d_total_gradient // onput
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_states * num_of_joints)
            return;

        int state_idx = idx / num_of_joints;
        int joint_idx = idx % num_of_joints;

        float sum = 0.0f;

        for (int i = 0; i < num_of_constraint_indexs; i++)
        {
            int constraint_idx = d_constraint_indexs[i];
            sum += d_gradient[num_of_states * num_of_joints * constraint_idx + state_idx * num_of_joints + joint_idx];
        }

        d_total_gradient[state_idx * num_of_joints + joint_idx] = sum;
    }

    __global__ void sum_error_kernel(
        int num_of_states, 
        int num_of_constraints,
        int * d_constraint_indexs, // [num_of_constraint_indexs]
        int num_of_constraint_indexs,
        float * d_costs, // [num_of_states * num_of_constraints]
        float * d_total_costs // onput
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_of_states)
            return;

        int state_idx = idx;

        float sum = 0.0f;

        for (int i = 0; i < num_of_constraint_indexs; i++)
        {
            int constraint_idx = d_constraint_indexs[i];
            sum += d_costs[num_of_states * constraint_idx + state_idx];
        }

        d_total_costs[state_idx] = sum;
    }

    void SingleArmStates::calculateTotalGradientAndError(const std::vector<int> & constraint_indexs)
    {
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states_ * num_of_joints + threadsPerBlock - 1) / threadsPerBlock;

        // allocate memory for constraint indexs
        int * d_constraint_indexs;
        size_t d_constraint_indexs_bytes = constraint_indexs.size() * sizeof(int);

        cudaMalloc(&d_constraint_indexs, d_constraint_indexs_bytes);
        cudaMemcpy(d_constraint_indexs, constraint_indexs.data(), d_constraint_indexs_bytes, cudaMemcpyHostToDevice);

        // Sum the gradients for each constraint
        sum_gradients_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            num_of_states_,
            num_of_joints,
            space_info_single_arm_space->num_of_constraints,
            d_constraint_indexs,
            constraint_indexs.size(),
            d_gradient,
            d_total_gradient
        );

        // // print total gradient
        // std::vector<float> total_gradient(num_of_states_ * num_of_joints, 0.0);
        // cudaMemcpy(total_gradient.data(), d_total_gradient, num_of_states_ * num_of_joints * sizeof(float), cudaMemcpyDeviceToHost);
        // std::cout << "Total Gradient: " << std::endl;
        // for (int i = 0; i < num_of_states_; i++)
        // {
        //     for (int j = 0; j < num_of_joints; j++)
        //     {
        //         std::cout << total_gradient[i * num_of_joints + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        sum_error_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            num_of_states_,
            space_info_single_arm_space->num_of_constraints,
            d_constraint_indexs,
            constraint_indexs.size(),
            d_costs,
            d_total_costs
        );

        // // print total error
        // std::vector<float> total_costs(num_of_states_, 0.0);
        // cudaMemcpy(total_costs.data(), d_total_costs, num_of_states_ * sizeof(float), cudaMemcpyDeviceToHost);
        // std::cout << "Total Costs: " << std::endl;
        // for (int i = 0; i < num_of_states_; i++)
        // {
        //     std::cout << total_costs[i] << std::endl;
        // }

        // synchronize
        CUDA_CHECK(cudaDeviceSynchronize());

        // free the memory
        cudaFree(d_constraint_indexs);
    }

    std::vector<std::vector<Eigen::Isometry3d>> SingleArmStates::getLinkPosesInBaseLinkHost() const
    {
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

        // Allocate memory for the link poses in base link frame
        std::vector<float> link_poses_in_base_link_flatten(num_of_states_ * space_info_single_arm_space->num_of_links * 4 * 4, 0.0);

        // Copy the link poses from device to host
        cudaMemcpy(link_poses_in_base_link_flatten.data(), d_link_poses_in_base_link, num_of_states_ * space_info_single_arm_space->num_of_links * 4 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

        // Reshape the link poses
        std::vector<std::vector<Eigen::Isometry3d>> link_poses_in_base_link(num_of_states_, std::vector<Eigen::Isometry3d>(space_info_single_arm_space->num_of_links));

        for (int i = 0; i < num_of_states_; i++)
        {
            for (int j = 0; j < space_info_single_arm_space->num_of_links; j++)
            {
                Eigen::Matrix4d M;
                for (int k = 0; k < 4; k++)
                {
                    for (int l = 0; l < 4; l++)
                    {
                        M(k, l) = link_poses_in_base_link_flatten[i * space_info_single_arm_space->num_of_links * 4 * 4 + j * 4 * 4 + k * 4 + l];
                    }
                }
                link_poses_in_base_link[i][j] = Eigen::Isometry3d(M);
            }
        }

        return link_poses_in_base_link;
    }

    std::vector<Eigen::Isometry3d> SingleArmStates::getLinkPoseInBaseLinkHost(std::string link_name) const
    {
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

        // get the index of the link
        int link_index = -1;
        for (int i = 0; i < space_info_single_arm_space->num_of_links; i++)
        {
            if (space_info_single_arm_space->link_names[i] == link_name)
            {
                link_index = i;
                break;
            }
        }
        if (link_index == -1)
        {
            throw std::runtime_error("Link " + link_name + " not found in the space");
        }

        std::vector<std::vector<Eigen::Isometry3d>> link_poses_in_base_link = getLinkPosesInBaseLinkHost();

        // Extract the link poses for the given link
        std::vector<Eigen::Isometry3d> result(num_of_states_);
        for (int i = 0; i < num_of_states_; i++)
        {
            result[i] = link_poses_in_base_link[i][link_index];
        }

        return result;
    }

    std::vector<Eigen::MatrixXd> SingleArmStates::getSpaceJacobianInBaseLinkHost(std::string link_name) const
    {
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

        // get the index of the link
        int link_index = -1;
        for (int i = 0; i < space_info_single_arm_space->num_of_links; i++)
        {
            if (space_info_single_arm_space->link_names[i] == link_name)
            {
                link_index = i;
                break;
            }
        }
        if (link_index == -1)
        {
            throw std::runtime_error("Link " + link_name + " not found in the space");
        }

        // Allocate memory for the space jacobian in base link frame
        std::vector<float> space_jacobian_in_base_link_flatten(num_of_states_ * space_info_single_arm_space->num_of_links * 6 * num_of_joints, 0.0);

        // Copy the space jacobian from device to host
        cudaMemcpy(space_jacobian_in_base_link_flatten.data(), d_space_jacobian_in_base_link, num_of_states_ * space_info_single_arm_space->num_of_links * 6 * num_of_joints * sizeof(float), cudaMemcpyDeviceToHost);

        // Reshape the space jacobian
        std::vector<Eigen::MatrixXd> space_jacobian_in_base_link(num_of_states_, Eigen::MatrixXd::Zero(num_of_joints, 6));

        for (int i = 0; i < num_of_states_; i++)
        {
            for (int j = 0; j < num_of_joints; j++)
            {
                for (int k = 0; k < 6; k++)
                {
                    space_jacobian_in_base_link[i](j, k) = space_jacobian_in_base_link_flatten[i * space_info_single_arm_space->num_of_links * 6 * num_of_joints + link_index * 6 * num_of_joints + j * 6 + k];
                }
            }
        }

        return space_jacobian_in_base_link;
    }


    void SingleArmStates::print() const
    {
        // static_cast the space_info to SingleArmSpaceInfo
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

        // Get the joint states
        std::vector<std::vector<float>> joint_states = getJointStatesFullHost();

        // print states name
        printf("SingleArmStates: \n");

        // Print the joint states
        for (int i = 0; i < num_of_states_; i++)
        {
            printf("State %d: ", i);
            for (int j = 0; j < num_of_joints; j++)
            {
                printf("%f ", joint_states[i][j]);
            }
            printf("\n");

            // // Get the link poses in base link frame
            // std::vector<std::vector<Eigen::Isometry3d>> link_poses_in_base_link = getLinkPosesInBaseLinkHost();
            // for (int j = 0; j < space_info_single_arm_space->num_of_links; j++)
            // {
            //     printf("Link %s pose in base link frame: \n", space_info_single_arm_space->link_names[j].c_str());
            //     // get the matrix
            //     auto link_matrix = link_poses_in_base_link[i][j].matrix();
            //     for (int k = 0; k < 4; k++)
            //     {
            //         for (int l = 0; l < 4; l++)
            //         {
            //             printf("%f ", link_matrix(k, l));
            //         }
            //         printf("\n");
            //     }
            // }
        }
    }

    SingleArmStateManager::~SingleArmStateManager()
    {
        if (num_of_states_ > 0)
        {
            cudaFree(d_joint_states);
            d_joint_states = nullptr;
        }
    }

    void SingleArmStateManager::clear()
    {
        if (num_of_states_ > 0)
        {
            // call the base class clear function
            BaseStateManager::clear();
            cudaFree(d_joint_states);
            d_joint_states = nullptr;
        }
    }

    std::vector<int> SingleArmStateManager::add_states(const BaseStatesPtr & states)
    {
        // static cast the states to SingleArmStates
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states);

        if (single_arm_states->getNumOfStates() == 0)
        {
            // return empty vector
            return std::vector<int>();
        }

        // get the data size of those new states
        size_t d_joint_states_bytes = single_arm_states->getNumOfStates() * single_arm_states->getNumOfJoints() * sizeof(float);

        if (num_of_states_ == 0) {
            // allocate memory for the states
            cudaMalloc(&d_joint_states, d_joint_states_bytes);

            // copy the data to the device
            cudaMemcpy(d_joint_states, single_arm_states->getJointStatesCuda(), d_joint_states_bytes, cudaMemcpyDeviceToDevice);

            // update the number of states
            num_of_states_ = single_arm_states->getNumOfStates();
            
            // return vector of 0 to num_of_states - 1
            return std::vector<int>(num_of_states_);
        }
        else {

            int old_num_of_states = num_of_states_;

            // manager's states is not empty, we need to extend the d_joint_states.
            size_t d_new_joint_states_bytes = (num_of_states_ + single_arm_states->getNumOfStates()) * num_of_joints * sizeof(float);

            float * d_new_joint_states;

            // allocate memory for the new states
            cudaMalloc(&d_new_joint_states, d_new_joint_states_bytes);

            // copy the old states to the new states
            cudaMemcpy(d_new_joint_states, d_joint_states, num_of_states_ * num_of_joints * sizeof(float), cudaMemcpyDeviceToDevice);
        
            // copy the new states to the new states
            cudaMemcpy(d_new_joint_states + num_of_states_ * num_of_joints, 
                single_arm_states->getJointStatesCuda(), 
                single_arm_states->getNumOfStates() * num_of_joints * sizeof(float), 
                cudaMemcpyDeviceToDevice);

            // free the old states
            cudaFree(d_joint_states);

            // update the states pointer
            d_joint_states = d_new_joint_states;

            // update the number of states
            num_of_states_ += single_arm_states->getNumOfStates();

            // return vector of num_of_states_ - single_arm_states->getNumOfStates() to num_of_states_ - 1
            std::vector<int> result(single_arm_states->getNumOfStates());
            for (int i = 0; i < single_arm_states->getNumOfStates(); i++)
            {
                result[i] = old_num_of_states + i;
            }
            return result;
        }

        // raise error if get here
        throw std::runtime_error("Error in SingleArmStateManager::add_states");
    }

    int SingleArmStateManager::find_k_nearest_neighbors(
        int k, const BaseStatesPtr & query_states, 
        const std::vector<std::vector<int>> & group_indexs,
        std::vector<std::vector<int>> & neighbors_index
    )
    {

        if (num_of_states_ == 0)
        {
            // raise error
            throw std::runtime_error("Error in SingleArmStateManager::find_k_nearest_neighbors: manager is empty");
        }
        if (query_states->getNumOfStates() == 0)
        {
            // raise error
            throw std::runtime_error("Error in SingleArmStateManager::find_k_nearest_neighbors: query states is empty");
        }
        // check if each group has at least one element
        for (size_t i = 0; i < group_indexs.size(); i++)
        {
            if (group_indexs[i].size() == 0)
            {
                throw std::runtime_error("Error in SingleArmStateManager::find_k_nearest_neighbors: group " + std::to_string(i) + " is empty");
            }
        }

        // static cast the query states to SingleArmStates
        SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info_);

        // static cast the states to SingleArmStates
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(query_states);
        float * d_query_joint_states = single_arm_states->getJointStatesCuda();

        neighbors_index.clear();

        int total_actual_k = 0;
        std::vector<int> actual_k_in_each_group;
        for (size_t i = 0; i < group_indexs.size(); i++)
        {
            actual_k_in_each_group.push_back((int)(group_indexs[i].size()) < k ? (int)(group_indexs[i].size()) : k);
            total_actual_k += actual_k_in_each_group[i];
        }

        float * d_distances_from_query_to_states;
        size_t d_distances_from_query_to_states_bytes = query_states->getNumOfStates() * num_of_states_ * sizeof(float);
        cudaMalloc(&d_distances_from_query_to_states, d_distances_from_query_to_states_bytes);

        // calculate the distance between the query states and the states in the manager
        int block_size = 256;
        int grid_size = (query_states->getNumOfStates() * num_of_states_ + block_size - 1) / block_size;

        calculate_joint_state_distance<<<grid_size, block_size>>>(
            d_query_joint_states, query_states->getNumOfStates(),
            d_joint_states, num_of_states_,
            num_of_joints, single_arm_space_info->d_active_joint_map, d_distances_from_query_to_states
        );

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        std::vector<std::vector<float>> distances_from_query_to_states(query_states->getNumOfStates(), std::vector<float>(num_of_states_));
        std::vector<float> distances_from_query_to_states_flatten(query_states->getNumOfStates() * num_of_states_);

        // copy the distances from device to host
        cudaMemcpy(distances_from_query_to_states_flatten.data(), d_distances_from_query_to_states, query_states->getNumOfStates() * num_of_states_ * sizeof(float), cudaMemcpyDeviceToHost);

        // reshape the distances
        for (int i = 0; i < query_states->getNumOfStates(); i++) {
            for (int j = 0; j < num_of_states_; j++) {
                distances_from_query_to_states[i][j] = distances_from_query_to_states_flatten[i * num_of_states_ + j];
            }
        }

        for (int i = 0; i < query_states->getNumOfStates(); i++) {
            std::vector<int> index_k_nearest_neighbors;
            for (size_t g = 0 ; g < group_indexs.size(); g++)
            {
                // find index of the k least distances of distances_from_query_to_states[i]
                std::vector<int> index_k_nearest_neighbors_of_group = kLeastIndices(distances_from_query_to_states[i], actual_k_in_each_group[g], group_indexs[g]);
                index_k_nearest_neighbors.insert(index_k_nearest_neighbors.end(), index_k_nearest_neighbors_of_group.begin(), index_k_nearest_neighbors_of_group.end());
            }

            neighbors_index.push_back(index_k_nearest_neighbors);
        }

        // free the memory
        cudaFree(d_distances_from_query_to_states);

        CUDA_CHECK(cudaGetLastError()); // Check for launch errors

        return total_actual_k;
    }

    // One block handles one query state for a given group.
    __global__ void find_the_nearest_neighbors_kernel(
        float * d_query_states_joint_values, int num_query_states,
        float * d_state_manager_states_joint_values,
        int * d_group_index_i, int num_of_group_index_i,
        int num_of_joints, int * d_active_joint_map, 
        int num_of_groups, int current_group_id,
        int * index_nearest_neighbor_of_each_group
    )
    {
        // Each block handles one query state.
        int query_idx = blockIdx.x;
        if (query_idx >= num_query_states) return;

        int query_state_base = query_idx * num_of_joints;
        int tid = threadIdx.x;
        // Initialize local best distance to a very high value.
        float best_distance_local = 1e30f;
        int best_index_local = -1;

        // Each thread processes a subset of candidate indices in a strided loop.
        for (int t = tid; t < num_of_group_index_i; t += blockDim.x)
        {
            int check_index = d_group_index_i[t];
            int manager_state_base = check_index * num_of_joints;
            float square_dis = 0.0f;
            for (int j = 0; j < num_of_joints; j++)
            {
                if (d_active_joint_map[j] != 0)
                {
                    float diff = d_query_states_joint_values[query_state_base + j] - 
                                d_state_manager_states_joint_values[manager_state_base + j];
                    square_dis += diff * diff;
                }
            }
            float dis = sqrtf(square_dis);
            if (dis < best_distance_local)
            {
                best_distance_local = dis;
                best_index_local = check_index;  // storing the candidate index in the group
            }
        }

        // Use shared memory to perform block-level reduction.
        // Allocate shared memory for both the best distances and corresponding indices.
        extern __shared__ char shared_mem[];
        float* shared_distance = (float*)shared_mem;
        int* shared_index = (int*)&shared_distance[blockDim.x];

        shared_distance[tid] = best_distance_local;
        shared_index[tid] = best_index_local;
        __syncthreads();

        // Reduction within the block.
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                if (shared_distance[tid + s] < shared_distance[tid])
                {
                    shared_distance[tid] = shared_distance[tid + s];
                    shared_index[tid] = shared_index[tid + s];
                }
            }
            __syncthreads();
        }

        // The first thread in the block writes the result.
        if (tid == 0)
        {
            index_nearest_neighbor_of_each_group[query_idx * num_of_groups + current_group_id] = shared_index[0];
        }
    }

    void SingleArmStateManager::find_the_nearest_neighbors(
        const BaseStatesPtr & query_states, 
        const std::vector<std::vector<int>> & group_indexs, 
        std::vector<std::vector<int>> & neighbors_index // output
    ) 
    {
        // Cast the query states to SingleArmStates.
        SingleArmStatesPtr query_single_arm_states = std::static_pointer_cast<SingleArmStates>(query_states);
        int num_query_states = query_single_arm_states->getNumOfStates();

        neighbors_index.clear();

        // static cast the query states to SingleArmStates
        SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info_);

        // Check for errors.
        if (num_of_states_ == 0)
        {
            throw std::runtime_error("Error in SingleArmStateManager::find_the_nearest_neighbors: manager is empty");
        }
        if (num_query_states == 0)
        {
            throw std::runtime_error("Error in SingleArmStateManager::find_the_nearest_neighbors: query states are empty");
        }

        // Verify that every group has at least one candidate.
        for (size_t i = 0; i < group_indexs.size(); i++)
        {
            if (group_indexs[i].empty())
            {
                throw std::runtime_error("Error in SingleArmStateManager::find_the_nearest_neighbors: group " + std::to_string(i) + " is empty");
            }
        }

        // Allocate device memory for each group's candidate indices.
        std::vector<int*> d_group_indexs(group_indexs.size());
        for (size_t i = 0; i < group_indexs.size(); i++)
        {
            size_t bytes = group_indexs[i].size() * sizeof(int);
            CUDA_CHECK(cudaMalloc(&d_group_indexs[i], bytes));
            CUDA_CHECK(cudaMemcpy(d_group_indexs[i], group_indexs[i].data(), bytes, cudaMemcpyHostToDevice));
        }

        // Allocate device memory for the output: one integer per query state per group.
        int * index_nearest_neighbor_of_each_group;
        CUDA_CHECK(cudaMalloc(&index_nearest_neighbor_of_each_group, group_indexs.size() * num_query_states * sizeof(int)));

        // Define kernel launch parameters.
        int threadsPerBlock = 256;               // Number of threads per block.
        int blocksPerGrid = num_query_states;      // One block per query state.
        int sharedMemSize = threadsPerBlock * (sizeof(float) + sizeof(int)); // Dynamic shared memory.

        // Launch the kernel for each group.
        for (size_t i = 0; i < group_indexs.size(); i++)
        {
            find_the_nearest_neighbors_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                query_single_arm_states->getJointStatesCuda(), num_query_states,
                d_joint_states,                  // Manager states (device pointer)
                d_group_indexs[i],               // Current group's candidate indices.
                group_indexs[i].size(),          // Number of candidates in the current group.
                num_of_joints,
                single_arm_space_info->d_active_joint_map,
                group_indexs.size(),             // Total number of groups.
                i,                             // Current group id.
                index_nearest_neighbor_of_each_group
            );
        }

        // Wait for all kernels to finish.
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy the flattened nearest neighbor indices from device to host.
        std::vector<int> flat_neighbors(num_query_states * group_indexs.size());
        CUDA_CHECK(cudaMemcpy(flat_neighbors.data(), index_nearest_neighbor_of_each_group, 
                num_query_states * group_indexs.size() * sizeof(int), cudaMemcpyDeviceToHost));

        // Reshape the flat vector into a 2D vector (each row corresponds to a query state).
        for (int i = 0; i < num_query_states; i++)
        {
            std::vector<int> neighbors_index_i;
            for (size_t j = 0; j < group_indexs.size(); j++)
            {
                neighbors_index_i.push_back(flat_neighbors[i * group_indexs.size() + j]);
            }
            neighbors_index.push_back(neighbors_index_i);
        }

        // Free all allocated device memory.
        for (size_t i = 0; i < group_indexs.size(); i++)
        {
            CUDA_CHECK(cudaFree(d_group_indexs[i]));
        }
        CUDA_CHECK(cudaFree(index_nearest_neighbor_of_each_group));
    }

    BaseStatesPtr SingleArmStateManager::get_states(const std::vector<int> & states_index)
    {
        // static cast the space_info to SingleArmSpaceInfo
        SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info_);

        // create a new SingleArmStates
        SingleArmStatesPtr extracted_states = std::make_shared<SingleArmStates>(states_index.size(), single_arm_space_info);

        float * d_extracted_joint_states = extracted_states->getJointStatesCuda();

        // copy the states from the manager to the extracted_states
        for (size_t i = 0; i < states_index.size(); i++)
        {
            // copy them asynchronously
            cudaMemcpyAsync(d_extracted_joint_states + i * num_of_joints, d_joint_states + states_index[i] * num_of_joints, num_of_joints * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        // wait for the copy to finish
        cudaDeviceSynchronize();

        return extracted_states;
    }

    BaseStatesPtr SingleArmStateManager::concatinate_states(const std::vector<BaseStatesPtr> & states)
    {
        // static cast the space_info to SingleArmSpaceInfo
        SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info_);

        int total_num_of_states = 0;
        for (size_t i = 0; i < states.size(); i++)
        {
            total_num_of_states += states[i]->getNumOfStates();
        }

        // create a new SingleArmStates
        SingleArmStatesPtr concatinated_states = std::make_shared<SingleArmStates>(total_num_of_states, single_arm_space_info);

        float * d_concatinated_joint_states = concatinated_states->getJointStatesCuda();

        // copy the states from the manager to the extracted_states
        int offset = 0;
        for (size_t i = 0; i < states.size(); i++)
        {
            SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(states[i]);
            int num_of_states_in_state_i = states[i]->getNumOfStates();
            // copy them asynchronously
            cudaMemcpyAsync(d_concatinated_joint_states + offset * num_of_joints, single_arm_states->getJointStatesCuda(), num_of_states_in_state_i * num_of_joints * sizeof(float), cudaMemcpyDeviceToDevice);
            offset += num_of_states_in_state_i;
        }

        // wait for the copy to finish
        cudaDeviceSynchronize();

        return concatinated_states;
    }
} // namespace CUDAMPLib
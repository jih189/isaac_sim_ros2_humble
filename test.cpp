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
        // C[12] = A[12] * B[0] + A[13] * B[4] + A[14] * B[8] + A[15] * B[12];
        // C[13] = A[12] * B[1] + A[13] * B[5] + A[14] * B[9] + A[15] * B[13];
        // C[14] = A[12] * B[2] + A[13] * B[6] + A[14] * B[10] + A[15] * B[14];
        // C[15] = A[12] * B[3] + A[13] * B[7] + A[14] * B[11] + A[15] * B[15];
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

    /**
        * @brief Get the rotation matrix from axis-angle representation
     */
    __device__ __forceinline__ void make_rotation_axis_angle(float angle, float x, float y, float z, float* R)
    {
        // Normalize the axis
        float length = sqrtf(x*x + y*y + z*z);
        if (length < 1e-12f)
        {
            // If axis is nearly zero-length, return identity
            set_identity(R);
            return;
        }

        x /= length;
        y /= length;
        z /= length;

        float c     = cosf(angle);
        float s     = sinf(angle);
        float one_c = 1.f - c;

        // Row-major rotation matrix
        R[0]  = c + x*x*one_c;     R[1]  = x*y*one_c - z*s;   R[2]  = x*z*one_c + y*s;    R[3]  = 0.f;
        R[4]  = y*x*one_c + z*s;   R[5]  = c + y*y*one_c;     R[6]  = y*z*one_c - x*s;    R[7]  = 0.f;
        R[8]  = z*x*one_c - y*s;   R[9]  = z*y*one_c + x*s;   R[10] = c + z*z*one_c;       R[11] = 0.f;
        R[12] = 0.f;               R[13] = 0.f;               R[14] = 0.f;                R[15] = 1.f;
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
                default:
                    printf("Unknown joint type: %d\n", joint_types[i]);
                    break;
            }
        }
    }



[cuda_test-1]       0       0       0       0       0       0       0
[cuda_test-1]   0.929       0       0       0       0       0       0
[cuda_test-1]       0  -0.812       0   -0.46       0 -0.1385       0
[cuda_test-1]       0       0       1       0       1       0       1
[cuda_test-1]       0       1       0       1       0       1       0
[cuda_test-1]       1       0       0       0       0       0       0

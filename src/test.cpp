#define CUDAMPLib_UNKNOWN 0
#define CUDAMPLib_REVOLUTE 1
#define CUDAMPLib_PRISMATIC 2
#define CUDAMPLib_PLANAR 3
#define CUDAMPLib_FLOATING 4
#define CUDAMPLib_FIXED 5

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

// given the joint values with size (num_of_joints * configuration_size)

int idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx >= configuration_size)
    return;

int tidx = threadIdx.x;
int base = blockIdx.x * blockDim.x;

extern __shared__ float joint_values_shared[]; // size is (num_of_joints * blockDim.x) * sizeof(float)
#pragma unroll
for (int i = 0; i < num_of_joints; i++)
{
    joint_values_shared[i * blockDim.x + tidx] = joint_values[base + i * blockDim.x + tidx]
}
__syncthreads();

if (idx == 0)
{
    // print the shared memory
    for (int i = 0; i < blockDim.x ; i++)
    {
        for (int j = 0; j < num_of_joints; j++)
        {
            std::cout << joint_values_shared[i * num_of_joints + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

#ifndef FLT_MAX
#define FLT_MAX __int_as_float(0x7f7fffff)    // 3.40282347e+38f
#endif

extern "C" {
    __device__ int startTreeCounter = 0;
    __device__ int goalTreeCounter = 0;
    __device__ int sampledCounter = 0;
}
__constant__ float joint_poses[480] = 
{
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, -0.124650, 0.000000, -0.000004, -1.000000, 0.238920, 0.000000, 1.000000, -0.000004, 0.311270, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, -0.213830, 0.000000, 1.000000, 0.000000, 0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, 0.213830, 0.000000, 1.000000, 0.000000, 0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, 0.001291, 0.000000, 1.000000, 0.000000, 0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, -0.000000, 0.235000, 0.000000, -1.000000, 0.000000, 0.000000, 0.000000, -0.000000, -1.000000, 0.287800, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, -0.213830, 0.000000, 1.000000, 0.000000, -0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, 0.213830, 0.000000, 1.000000, 0.000000, -0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, 0.001291, 0.000000, 1.000000, 0.000000, -0.187380, -0.000000, -0.000000, 1.000000, 0.055325, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, -0.086875, 0.000000, 1.000000, 0.000000, 0.000000, -0.000000, -0.000000, 1.000000, 0.377425, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, -0.000000, 0.000000, -0.086875, 0.000000, 1.000000, 0.000000, 0.000000, -0.000000, -0.000000, 1.000000, 0.377430, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.053125, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.603001, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.142530, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.057999, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.055000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.022500, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.045000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
0.000000, 0.000000, 1.000000, 0.000000, -1.000000, 0.000000, 0.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.020000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
0.000000, 0.000000, 1.000000, 0.000000, -1.000000, 0.000000, 0.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.119525, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.348580, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.117000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.060000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.219000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.133000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.197000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.124500, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.138500, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.166450, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, -0.015425, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 
1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.015425, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000
};
__constant__ float joint_axes[90] = 
{
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 1.000000, 
0.000000, 0.000000, -1.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 1.000000, 
0.000000, 1.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 1.000000, 
0.000000, 1.000000, 0.000000, 
1.000000, 0.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
1.000000, 0.000000, 0.000000, 
0.000000, 1.000000, 0.000000, 
1.000000, 0.000000, 0.000000, 
0.000000, 0.000000, 0.000000, 
0.000000, -1.000000, 0.000000, 
0.000000, 1.000000, 0.000000
};

 // Multiply two 4x4 matrices (row-major order)
__device__ __forceinline__ void multiply4x4(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            float* __restrict__ C)
{
    C[0] = A[0] * B[0] + A[1] * B[4] + A[2] * B[8]  + A[3] * B[12];
    C[1] = A[0] * B[1] + A[1] * B[5] + A[2] * B[9]  + A[3] * B[13];
    C[2] = A[0] * B[2] + A[1] * B[6] + A[2] * B[10] + A[3] * B[14];
    C[3] = A[0] * B[3] + A[1] * B[7] + A[2] * B[11] + A[3] * B[15];

    C[4] = A[4] * B[0] + A[5] * B[4] + A[6] * B[8]  + A[7] * B[12];
    C[5] = A[4] * B[1] + A[5] * B[5] + A[6] * B[9]  + A[7] * B[13];
    C[6] = A[4] * B[2] + A[5] * B[6] + A[6] * B[10] + A[7] * B[14];
    C[7] = A[4] * B[3] + A[5] * B[7] + A[6] * B[11] + A[7] * B[15];

    C[8]  = A[8]  * B[0] + A[9]  * B[4] + A[10] * B[8]  + A[11] * B[12];
    C[9]  = A[8]  * B[1] + A[9]  * B[5] + A[10] * B[9]  + A[11] * B[13];
    C[10] = A[8]  * B[2] + A[9]  * B[6] + A[10] * B[10] + A[11] * B[14];
    C[11] = A[8]  * B[3] + A[9]  * B[7] + A[10] * B[11] + A[11] * B[15];

    // Last row is fixed as [0, 0, 0, 1]
    C[12] = 0.f; C[13] = 0.f; C[14] = 0.f; C[15] = 1.f;
}

// Fixed joint: multiply parent's pose with joint's fixed pose.
__device__ __forceinline__ void fixed_joint_fn_cuda(const float* parent_link_pose,
                                                    const float* joint_pose,
                                                    float* link_pose)
{
    multiply4x4(parent_link_pose, joint_pose, link_pose);
}

// Create a rotation matrix from an axis-angle representation.
__device__ __forceinline__ void make_rotation_axis_angle(float angle, float x, float y, float z, float* R)
{
    float length = sqrtf(x * x + y * y + z * z);
    const float thresh = 1e-12f;
    float valid = (length >= thresh) ? 1.f : 0.f;
    float inv_length = 1.f / fmaxf(length, thresh);
    float nx = x * inv_length * valid;
    float ny = y * inv_length * valid;
    float nz = z * inv_length * valid;
    float c = cosf(angle);
    float s = sinf(angle);
    float one_c = 1.f - c;

    float r0  = c + nx * nx * one_c;
    float r1  = nx * ny * one_c - nz * s;
    float r2  = nx * nz * one_c + ny * s;
    float r4  = ny * nx * one_c + nz * s;
    float r5  = c + ny * ny * one_c;
    float r6  = ny * nz * one_c - nx * s;
    float r8  = nz * nx * one_c - ny * s;
    float r9  = nz * ny * one_c + nx * s;
    float r10 = c + nz * nz * one_c;

    R[0]  = r0 * valid + (1.f - valid) * 1.f; R[1]  = r1 * valid;           R[2]  = r2 * valid;           R[3]  = 0.f;
    R[4]  = r4 * valid;           R[5]  = r5 * valid + (1.f - valid) * 1.f; R[6]  = r6 * valid;           R[7]  = 0.f;
    R[8]  = r8 * valid;           R[9]  = r9 * valid;           R[10] = r10 * valid + (1.f - valid) * 1.f; R[11] = 0.f;
    R[12] = 0.f; R[13] = 0.f; R[14] = 0.f; R[15] = 1.f;
}

// Revolute joint: compute rotation transformation then multiply with parent's pose.
__device__ __forceinline__ void revolute_joint_fn_cuda(const float* parent_link_pose,
                                                        const float* joint_pose,
                                                        const float* joint_axis,
                                                        float joint_value,
                                                        float* link_pose)
{
    float joint_transform[16];
    make_rotation_axis_angle(joint_value, joint_axis[0], joint_axis[1], joint_axis[2], joint_transform);
    
    float temp[16];
    multiply4x4(parent_link_pose, joint_pose, temp);
    multiply4x4(temp, joint_transform, link_pose);
}

// Prismatic joint: create a translation matrix and combine with parent's pose.
__device__ __forceinline__ void prism_joint_fn_cuda(const float* parent_link_pose,
                                                    const float* joint_pose,
                                                    const float* joint_axis,
                                                    float joint_value,
                                                    float* link_pose)
{
    float x = joint_axis[0], y = joint_axis[1], z = joint_axis[2];
    float T[16] = {
        1.0f, 0.0f, 0.0f, x * joint_value,
        0.0f, 1.0f, 0.0f, y * joint_value,
        0.0f, 0.0f, 1.0f, z * joint_value,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    float joint_pose_T[16];
    multiply4x4(joint_pose, T, joint_pose_T);
    multiply4x4(parent_link_pose, joint_pose_T, link_pose);
}       

__device__ __forceinline__ void kin_forward(float * configuration, float * self_collision_spheres)
{
    // based on the default value and fixed joint, calculate the full joint values.
    float full_joint_values[30];
    full_joint_values[0] = 0.000000;
    full_joint_values[1] = 0.000000;
    full_joint_values[2] = 0.000000;
    full_joint_values[3] = 0.000000;
    full_joint_values[4] = 0.000000;
    full_joint_values[5] = 0.000000;
    full_joint_values[6] = 0.000000;
    full_joint_values[7] = 0.000000;
    full_joint_values[8] = 0.000000;
    full_joint_values[9] = 0.000000;
    full_joint_values[10] = 0.000000;
    full_joint_values[11] = 0.000000;
    full_joint_values[12] = 0.000000;
    full_joint_values[13] = 0.000000;
    full_joint_values[14] = 0.000000;
    full_joint_values[15] = 0.000000;
    full_joint_values[16] = 0.000000;
    full_joint_values[17] = 0.000000;
    full_joint_values[18] = 0.000000;
    full_joint_values[19] = 0.000000;
    full_joint_values[20] = configuration[0];
    full_joint_values[21] = configuration[1];
    full_joint_values[22] = configuration[2];
    full_joint_values[23] = configuration[3];
    full_joint_values[24] = configuration[4];
    full_joint_values[25] = configuration[5];
    full_joint_values[26] = configuration[6];
    full_joint_values[27] = 0.000000;
    full_joint_values[28] = 0.000000;
    full_joint_values[29] = 0.000000;
    float link_poses[480];
    // set the base link pose to identity
    link_poses[0] = 1.0f; link_poses[1] = 0.0f; link_poses[2] = 0.0f; link_poses[3] = 0.0f;
    link_poses[4] = 0.0f; link_poses[5] = 1.0f; link_poses[6] = 0.0f; link_poses[7] = 0.0f;
    link_poses[8] = 0.0f; link_poses[9] = 0.0f; link_poses[10] = 1.0f; link_poses[11] = 0.0f;
    link_poses[12] = 0.0f; link_poses[13] = 0.0f; link_poses[14] = 0.0f; link_poses[15] = 1.0f;
    // Unrolled joint 1
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[0], 
        &joint_poses[16], 
        &link_poses[32] 
    );
    // Unrolled joint 2
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[32], 
        &joint_axes[6], 
        full_joint_values[2], 
        &link_poses[48] 
    );
    // Unrolled joint 3
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[48], 
        &joint_axes[9], 
        full_joint_values[3], 
        &link_poses[64] 
    );
    // Unrolled joint 4
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[64], 
        &joint_axes[12], 
        full_joint_values[4], 
        &link_poses[80] 
    );
    // Unrolled joint 5
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[0], 
        &joint_poses[80], 
        &link_poses[96] 
    );
    // Unrolled joint 6
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[96], 
        &joint_axes[18], 
        full_joint_values[6], 
        &link_poses[112] 
    );
    // Unrolled joint 7
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[112], 
        &joint_axes[21], 
        full_joint_values[7], 
        &link_poses[128] 
    );
    // Unrolled joint 8
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[0], 
        &joint_poses[128], 
        &joint_axes[24], 
        full_joint_values[8], 
        &link_poses[144] 
    );
    // Unrolled joint 9
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[0], 
        &joint_poses[144], 
        &link_poses[160] 
    );
    // Unrolled joint 10
    // prismatic joint
    prism_joint_fn_cuda( &link_poses[0], 
        &joint_poses[160], 
        &joint_axes[30], 
        full_joint_values[10], 
        &link_poses[176] 
    );
    // Unrolled joint 11
    // prismatic joint
    prism_joint_fn_cuda( &link_poses[160], 
        &joint_poses[176], 
        &joint_axes[33], 
        full_joint_values[11], 
        &link_poses[192] 
    );
    // Unrolled joint 12
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[160], 
        &joint_poses[192], 
        &link_poses[208] 
    );
    // Unrolled joint 13
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[160], 
        &joint_poses[208], 
        &joint_axes[39], 
        full_joint_values[13], 
        &link_poses[224] 
    );
    // Unrolled joint 14
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[208], 
        &joint_poses[224], 
        &joint_axes[42], 
        full_joint_values[14], 
        &link_poses[240] 
    );
    // Unrolled joint 15
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[224], 
        &joint_poses[240], 
        &link_poses[256] 
    );
    // Unrolled joint 16
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[240], 
        &joint_poses[256], 
        &link_poses[272] 
    );
    // Unrolled joint 17
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[256], 
        &joint_poses[272], 
        &link_poses[288] 
    );
    // Unrolled joint 18
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[240], 
        &joint_poses[288], 
        &link_poses[304] 
    );
    // Unrolled joint 19
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[288], 
        &joint_poses[304], 
        &link_poses[320] 
    );
    // Unrolled joint 20
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[160], 
        &joint_poses[320], 
        &joint_axes[60], 
        full_joint_values[20], 
        &link_poses[336] 
    );
    // Unrolled joint 21
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[320], 
        &joint_poses[336], 
        &joint_axes[63], 
        full_joint_values[21], 
        &link_poses[352] 
    );
    // Unrolled joint 22
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[336], 
        &joint_poses[352], 
        &joint_axes[66], 
        full_joint_values[22], 
        &link_poses[368] 
    );
    // Unrolled joint 23
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[352], 
        &joint_poses[368], 
        &joint_axes[69], 
        full_joint_values[23], 
        &link_poses[384] 
    );
    // Unrolled joint 24
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[368], 
        &joint_poses[384], 
        &joint_axes[72], 
        full_joint_values[24], 
        &link_poses[400] 
    );
    // Unrolled joint 25
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[384], 
        &joint_poses[400], 
        &joint_axes[75], 
        full_joint_values[25], 
        &link_poses[416] 
    );
    // Unrolled joint 26
    // revolute joint
    revolute_joint_fn_cuda( &link_poses[400], 
        &joint_poses[416], 
        &joint_axes[78], 
        full_joint_values[26], 
        &link_poses[432] 
    );
    // Unrolled joint 27
    // fixed joint
    fixed_joint_fn_cuda( &link_poses[416], 
        &joint_poses[432], 
        &link_poses[448] 
    );
    // Unrolled joint 28
    // prismatic joint
    prism_joint_fn_cuda( &link_poses[432], 
        &joint_poses[448], 
        &joint_axes[84], 
        full_joint_values[28], 
        &link_poses[464] 
    );
    // Unrolled joint 29
    // prismatic joint
    prism_joint_fn_cuda( &link_poses[432], 
        &joint_poses[464], 
        &joint_axes[87], 
        full_joint_values[29], 
        &link_poses[480] 
    );


}
        
extern "C" __global__ void cRRTCKernel(float * d_start_tree_configurations, float * d_goal_tree_configurations, int * d_start_tree_parent_indexs, int * d_goal_tree_parent_indexs, float * d_sampled_configurations) {
    __shared__ float * tree_to_expand;
    __shared__ int * tree_to_expand_parent_indexs;
    __shared__ int localTargetTreeCounter;
    __shared__ int localSampledCounter;
    __shared__ int localStartTreeCounter;
    __shared__ int localGoalTreeCounter;
    __shared__ float partial_distance_cost_from_nn[32];
    __shared__ int partial_nn_index[32];
    __shared__ float local_sampled_configuration[7];
    __shared__ float local_parent_configuration[7];
    __shared__ float local_delta_motion[7];
    __shared__ int local_parent_index;
    __shared__ float local_nearest_neighbor_distance;
    __shared__ float local_motion_configurations[224]; 
    __shared__ int motion_step;
    const int tid = threadIdx.x;
    // run for loop with max_interations_ iterations
    for (int i = 0; i < 1; i++) {

        // Need to decide which tree to expand based on their sizes. The smaller tree will be expanded.
        if (tid == 0)
        {
            // increase the sampledCounter with atomic operation
            localSampledCounter = atomicAdd(&sampledCounter, 1);
            localStartTreeCounter = startTreeCounter;
            localGoalTreeCounter = goalTreeCounter;

            if (localStartTreeCounter < localGoalTreeCounter) {
                tree_to_expand = d_start_tree_configurations;
                tree_to_expand_parent_indexs = d_start_tree_parent_indexs;
                localTargetTreeCounter = localStartTreeCounter;
            } else {
                tree_to_expand = d_goal_tree_configurations;
                tree_to_expand_parent_indexs = d_goal_tree_parent_indexs;
                localTargetTreeCounter = localGoalTreeCounter;
            }
        }

        __syncthreads();
        if (localSampledCounter >= 1)
            return; // meet the max_iteration, then stop the block.
        if(tid == 0) {
            printf("localStartTreeCounter: %d\n", localStartTreeCounter);
            printf("localGoalTreeCounter: %d\n", localGoalTreeCounter);
            printf("localSampledCounter: %d\n", localSampledCounter);
            printf("Sampled configuration: ");
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 0]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 1]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 2]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 3]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 4]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 5]);
            printf("%f ", d_sampled_configurations[localSampledCounter * 7 + 6]);
            printf("\n");
        }
        // Load the sampled configuration into shared memory
        if (tid < 7) {
            local_sampled_configuration[tid] = d_sampled_configurations[localSampledCounter * 7 + tid];
        }
        __syncthreads();

        // Find the nearest configuration in the tree_to_expand to the sampled configuration with reduction operation

        float best_dist = FLT_MAX;
        int best_index = -1;
        for (int j = tid; j < localTargetTreeCounter; j += blockDim.x){
            float dist = 0.0f;
            float diff = 0.0f;
            diff = tree_to_expand[j * 7 + 0] - local_sampled_configuration[0];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 1] - local_sampled_configuration[1];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 2] - local_sampled_configuration[2];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 3] - local_sampled_configuration[3];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 4] - local_sampled_configuration[4];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 5] - local_sampled_configuration[5];
            dist += diff * diff;
            diff = tree_to_expand[j * 7 + 6] - local_sampled_configuration[6];
            dist += diff * diff;

            if (dist < best_dist) {
                best_dist = dist;
                best_index = j;
            }
        }

        // Write the local best distance and index to the shared memory
        partial_distance_cost_from_nn[tid] = best_dist;
        partial_nn_index[tid] = best_index;
        __syncthreads();

        // Perform reduction to find the best distance and index
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                if (partial_distance_cost_from_nn[tid + stride] < partial_distance_cost_from_nn[tid]) {
                    partial_distance_cost_from_nn[tid] = partial_distance_cost_from_nn[tid + stride];
                    partial_nn_index[tid] = partial_nn_index[tid + stride];
                }
            }
            __syncthreads();
        }

        // After the reduction, thread 0 has the overall nearest neighbor's index and its squared distance.
        if (tid == 0) {
            local_nearest_neighbor_distance = sqrtf(partial_distance_cost_from_nn[0]);
            local_parent_index = partial_nn_index[0];
            motion_step = min((int)(local_nearest_neighbor_distance / 0.020000), 32);
            printf("Nearest neighbor index: %d, Euclidean distance: %f motion step: %d \n ", local_parent_index, local_nearest_neighbor_distance, motion_step);
        }
        __syncthreads();
        // Calculate the delta motion from the nearest configuration to the sampled configuration
        if (tid < 7) {
            local_parent_configuration[tid] = tree_to_expand[local_parent_index * 7 + tid];
            local_delta_motion[tid] = (local_sampled_configuration[tid] - local_parent_configuration[tid]) / local_nearest_neighbor_distance * 0.020000;
        }

        __syncthreads();
        // interpolate the new configuration from the nearest configuration and the sampled configuration
        for (int j = tid; j < 7 * motion_step; j += blockDim.x) {
            int state_ind_in_motion = j / 7;
            int joint_ind_in_state = j % 7;
            local_motion_configurations[j] = local_parent_configuration[joint_ind_in_state] + local_delta_motion[joint_ind_in_state] * state_ind_in_motion;
        }
        __syncthreads();
    }

}
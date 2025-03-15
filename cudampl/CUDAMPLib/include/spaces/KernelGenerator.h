#pragma once

#include <string>

namespace CUDAMPLib
{
    inline std::string genForwardKinematicsKernelCode(
        std::vector<int> joint_types, 
        const int num_of_links, 
        const int num_of_joints,
        std::vector<float> joint_poses_flatten, 
        std::vector<float> joint_axes_flatten, 
        std::vector<int> link_parent_link_maps
    )
    {
        std::string kernel_code;

        // set constants
        kernel_code += "__constant__ float joint_poses[" + std::to_string(num_of_joints * 16) + "] = {";
        for (size_t i = 0; i < joint_poses_flatten.size(); ++i)
        {
            kernel_code += std::to_string(joint_poses_flatten[i]);
            if (i < joint_poses_flatten.size() - 1)
                kernel_code += ", ";
        }
        kernel_code += "};\n";

        kernel_code += "__constant__ float joint_axes[" + std::to_string(num_of_joints * 3) + "] = {";
        for (size_t i = 0; i < joint_axes_flatten.size(); ++i)
        {
            kernel_code += std::to_string(joint_axes_flatten[i]);
            if (i < joint_axes_flatten.size() - 1)
                kernel_code += ", ";
        }
        kernel_code += "};\n";

        kernel_code += R"(

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

//---------------------------------------------------------------------
// Kernel function for forward kinematics (nvrtc)
//---------------------------------------------------------------------
extern "C" __global__ 
void kin_forward_nvrtc_kernel(
    const float* __restrict__ joint_values,
    const int configuration_size,
    float* __restrict__ link_poses_set)
{
    extern __shared__ float joint_values_shared[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidx = threadIdx.x;
)";
                    kernel_code += "    int base = blockIdx.x * blockDim.x * " + std::to_string(num_of_joints) + ";\n";
                    kernel_code += "    #pragma unroll\n";
                    kernel_code += "    for (int i = 0; i < " + std::to_string(num_of_joints) + "; i++)\n";
                    kernel_code += "    {\n";
                    kernel_code += "        joint_values_shared[i * blockDim.x + tidx] = joint_values[base + i * blockDim.x + tidx];\n";
                    kernel_code += "    }\n";

                    kernel_code += "    __syncthreads();\n\n";
                    kernel_code += "    if (idx >= configuration_size) {return;}\n";
                    
                    kernel_code += "    float* current_link_pose_0 = &link_poses_set[idx * " + std::to_string(num_of_links * 16) + "];\n";
                    kernel_code += "    current_link_pose_0[0] = 1.0; current_link_pose_0[1] = 0.0; current_link_pose_0[2] = 0.0; current_link_pose_0[3] = 0.0;\n";
                    kernel_code += "    current_link_pose_0[4] = 0.0; current_link_pose_0[5] = 1.0; current_link_pose_0[6] = 0.0; current_link_pose_0[7] = 0.0;\n";
                    kernel_code += "    current_link_pose_0[8] = 0.0; current_link_pose_0[9] = 0.0; current_link_pose_0[10] = 1.0; current_link_pose_0[11] = 0.0;\n";
                    kernel_code += "    current_link_pose_0[12] = 0.0; current_link_pose_0[13] = 0.0; current_link_pose_0[14] = 0.0; current_link_pose_0[15] = 1.0;\n\n";
                    
                    for (size_t i = 1; i < joint_types.size(); ++i)
                    {
                        // Start an unrolled block for joint i.
                        kernel_code += "    // Unrolled joint " + std::to_string(i) + "\n";
                        // kernel_code += "    float link_pose_" + std::to_string(i) + "[16];\n";
                        kernel_code += "    float* current_link_pose_" + std::to_string(i) + " = &link_poses_set[idx * " + std::to_string(num_of_links * 16) + " + " + std::to_string(i * 16) + "];\n";

                        // Depending on the joint type, insert the corresponding call.
                        int type = joint_types[i];
                        if (type == 1)  // REVOLUTE
                        {
                            kernel_code += "    revolute_joint_fn_cuda(current_link_pose_" + std::to_string(link_parent_link_maps[i]) +
                                ", &joint_poses[" + std::to_string(i * 16) + "], &joint_axes[" + std::to_string(i * 3) + "], joint_values_shared[tidx * " + std::to_string(num_of_joints) + " + " + std::to_string(i) +
                                "], current_link_pose_" + std::to_string(i) + ");\n";
                        }
                        else if (type == 2)  // PRISMATIC
                        {
                            kernel_code += "    prism_joint_fn_cuda(current_link_pose_" + std::to_string(link_parent_link_maps[i]) +
                                ", &joint_poses[" + std::to_string(i * 16) + "], &joint_axes[" + std::to_string(i * 3) + "], joint_values_shared[tidx * " + std::to_string(num_of_joints) + " + " + std::to_string(i) +
                                "], current_link_pose_" + std::to_string(i) + ");\n";
                        }
                        else if (type == 5)  // FIXED
                        {
                            kernel_code += "    fixed_joint_fn_cuda(current_link_pose_" + std::to_string(link_parent_link_maps[i]) +
                                ", &joint_poses[" + std::to_string(i * 16) + "], current_link_pose_" + std::to_string(i) + ");\n";
                        }
                        else
                        {
                            kernel_code += "    // Unsupported joint type: " + std::to_string(type) + "\n";
                        }
                        kernel_code += "\n";
                    }

                    // Close the kernel function and extern "C" block.
                    kernel_code += R"(})";

        return kernel_code;
    }
} // namespace CUDAMPLib
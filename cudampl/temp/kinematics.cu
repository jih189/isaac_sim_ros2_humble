#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014
#include "kinematics.h"
#include <cuda_runtime.h>

__device__ void multiply4x4(const float* A, const float* B, float* C)
{
    for(int row = 0; row < 4; ++row)
    {
        for(int col = 0; col < 4; ++col)
        {
            C[row * 4 + col] = 0.f;
            for(int k = 0; k < 4; ++k)
            {
                C[row * 4 + col] += A[row * 4 + k] * B[k * 4 + col];
            }
        }
    }
}

__device__ void fixed_joint_fn_cuda(
    float* parent_link_pose,
    float* joint_pose,
    float* link_pose
)
{
    // link_pose = parent_link_pose * joint_pose
    multiply4x4(parent_link_pose, joint_pose, link_pose);
}

__device__ void set_identity(float* M)
{
    // Row-major identity
    // 1 0 0 0
    // 0 1 0 0
    // 0 0 1 0
    // 0 0 0 1
    M[0]  = 1.f;  M[1]  = 0.f;  M[2]  = 0.f;  M[3]  = 0.f;
    M[4]  = 0.f;  M[5]  = 1.f;  M[6]  = 0.f;  M[7]  = 0.f;
    M[8]  = 0.f;  M[9]  = 0.f;  M[10] = 1.f;  M[11] = 0.f;
    M[12] = 0.f;  M[13] = 0.f;  M[14] = 0.f;  M[15] = 1.f;
}

__device__ void make_rotation_axis_angle(float angle, float x, float y, float z, float* R)
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

__device__ void revolute_joint_fn_cuda(
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

__device__ void prism_joint_fn_cuda(
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
    // for (int i = 0; i < 4; ++i)
    // {
    //     for (int j = 0; j < 4; ++j)
    //     {
    //         float sum = 0.0f;
    //         for (int k = 0; k < 4; ++k)
    //         {
    //             sum += joint_pose[i * 4 + k] * T[k * 4 + j];
    //         }
    //         joint_pose_T[i * 4 + j] = sum;
    //     }
    // }
    multiply4x4(joint_pose, T, joint_pose_T);

    //------------------------------------------------------------------------------
    // 3) Multiply parent_link_pose * joint_pose_T -> final link_pose
    //------------------------------------------------------------------------------
    // for (int i = 0; i < 4; ++i)
    // {
    //     for (int j = 0; j < 4; ++j)
    //     {
    //         float sum = 0.0f;
    //         for (int k = 0; k < 4; ++k)
    //         {
    //             sum += parent_link_pose[i * 4 + k] * joint_pose_T[k * 4 + j];
    //         }
    //         link_pose[i * 4 + j] = sum;
    //     }
    // }
    multiply4x4(parent_link_pose, joint_pose_T, link_pose);
}

__global__ void kin_forward_kernel(
    float* joint_values, 
    int num_of_joint,
    int configuration_size,
    int* joint_types,
    float* joint_poses,
    int num_of_links,
    float* joint_axes,
    int* link_maps,
    float* link_poses_set
) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < configuration_size) {

        // set the first link pose to identity matrix because it is the base link
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                link_poses_set[idx * num_of_links * 16 + i * 4 + j] = 0.0f;
            }
            link_poses_set[idx * num_of_links * 16 + i * 4 + i] = 1.0f;
        }

        // Calculate forward kinematics for each link
        size_t j = 0;
        for (size_t i = 1; i < num_of_links; i++) // The first link is the base link, so we can skip it
        {
            float* parent_link_pose = &link_poses_set[idx * num_of_links * 16 + link_maps[i] * 16];
            float* current_link_pose = &link_poses_set[idx * num_of_links * 16 + i * 16];
            // based on the joint type, calculate the link pose
            switch (joint_types[i])
            {
                case CUDAMPLib_REVOLUTE:
                    revolute_joint_fn_cuda(parent_link_pose, &joint_poses[i * 16], &joint_axes[i * 3], joint_values[idx * num_of_joint + j], current_link_pose);
                    j++;
                    break;
                case CUDAMPLib_PRISMATIC:
                    prism_joint_fn_cuda(parent_link_pose, &joint_poses[i * 16], &joint_axes[i * 3], joint_values[idx * num_of_joint + j], current_link_pose);
                    j++;
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
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                link_poses_set[idx * num_of_links * 16 + i * 4 + j] = 0.0f;
            }
            link_poses_set[idx * num_of_links * 16 + i * 4 + i] = 1.0f;
        }

        // Calculate forward kinematics for each link
        size_t j = 0;
        for (size_t i = 1; i < num_of_links; i++) // The first link is the base link, so we can skip it
        {
            float* parent_link_pose = &link_poses_set[idx * num_of_links * 16 + link_maps[i] * 16];
            float* current_link_pose = &link_poses_set[idx * num_of_links * 16 + i * 16];
            // based on the joint type, calculate the link pose
            switch (joint_types[i])
            {
                case CUDAMPLib_REVOLUTE:
                    revolute_joint_fn_cuda(parent_link_pose, &joint_poses[i * 16], &joint_axes[i * 3], joint_values[idx * num_of_joint + j], current_link_pose);
                    j++;
                    break;
                case CUDAMPLib_PRISMATIC:
                    prism_joint_fn_cuda(parent_link_pose, &joint_poses[i * 16], &joint_axes[i * 3], joint_values[idx * num_of_joint + j], current_link_pose);
                    j++;
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

void CUDAMPLib::kin_forward_cuda(
    const std::vector<std::vector<float>>& joint_values,
    const std::vector<int>& joint_types,
    const std::vector<Eigen::Isometry3d>& joint_poses,
    const std::vector<Eigen::Vector3d>& joint_axes,
    const std::vector<int>& link_maps,
    std::vector<std::vector<Eigen::Isometry3d>>& link_poses_set)
{
    if (joint_values.size() == 0)
    {
        std::cout << "No joint values provided." << std::endl;
        return;
    }
    
    // Prepare cuda memory
    int num_of_joints = joint_values[0].size();
    int num_of_links = link_maps.size();
    int num_of_config = joint_values.size();
    int joint_values_size = num_of_config * num_of_joints;
    int joint_values_bytes = joint_values_size * sizeof(float);
    int joint_types_bytes = joint_types.size() * sizeof(int);
    int size_of_pose_matrix = 4 * 4 * sizeof(float); // We do not need the last row of the matrix
    int joint_poses_bytes = joint_poses.size() * size_of_pose_matrix;
    int joint_axes_bytes = joint_axes.size() * sizeof(float) * 3;
    int link_maps_bytes = link_maps.size() * sizeof(int);
    int link_poses_set_size = num_of_links * num_of_config * size_of_pose_matrix;
    int link_poses_set_bytes = link_poses_set_size * sizeof(float);

    // Allocate device memory
    float *d_joint_values;
    int *d_joint_types;
    float *d_joint_poses;
    float *d_joint_axes;
    int *d_link_maps;
    float *d_link_poses_set;

    cudaMalloc(&d_joint_values, joint_values_bytes);
    cudaMalloc(&d_joint_types, joint_types_bytes);
    cudaMalloc(&d_joint_poses, joint_poses_bytes);
    cudaMalloc(&d_joint_axes, joint_axes_bytes);
    cudaMalloc(&d_link_maps, link_maps_bytes);
    cudaMalloc(&d_link_poses_set, link_poses_set_bytes);

    // Copy data from host to device
    cudaMemcpy(d_joint_values, floatVectorFlatten(joint_values).data(), joint_values_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_joint_types, joint_types.data(), joint_types_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_joint_poses, IsometryVectorFlatten(joint_poses).data(), joint_poses_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_joint_axes, Vector3dflatten(joint_axes).data(), joint_axes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_link_maps, link_maps.data(), link_maps_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_of_config + threadsPerBlock - 1) / threadsPerBlock;

    kin_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_joint_values, 
        num_of_joints,
        num_of_config,
        d_joint_types,
        d_joint_poses,
        num_of_links,
        d_joint_axes,
        d_link_maps,
        d_link_poses_set
    );
    cudaDeviceSynchronize();

    std::vector<float> h_link_poses_set(link_poses_set_size);
    cudaMemcpy(h_link_poses_set.data(), d_link_poses_set, link_poses_set_bytes, cudaMemcpyDeviceToHost);

    link_poses_set.clear();
    for (int i = 0; i < num_of_config; i++)
    {
        link_poses_set.push_back(fromFloatVectorToIsometry3d(std::vector<float>(h_link_poses_set.begin() + i * num_of_links * 16, h_link_poses_set.begin() + (i + 1) * num_of_links * 16)));
    }

    // Free device memory
    cudaFree(d_joint_values);
    cudaFree(d_joint_types);
    cudaFree(d_joint_poses);
    cudaFree(d_joint_axes);
    cudaFree(d_link_maps);
    cudaFree(d_link_poses_set);
}

void CUDAMPLib::kin_forward_collision_spheres_cuda(
    const std::vector<std::vector<float>>& joint_values,
    const std::vector<int>& joint_types,
    const std::vector<Eigen::Isometry3d>& joint_poses,
    const std::vector<Eigen::Vector3d>& joint_axes,
    const std::vector<int>& link_maps,
    const std::vector<int>& collision_spheres_map,
    const std::vector<std::vector<float>>& collision_spheres_pos,
    std::vector<std::vector<Eigen::Isometry3d>>& link_poses_set,
    std::vector<std::vector<std::vector<float>>>& collision_spheres_pos_in_baselink
)
{
    if (joint_values.size() == 0)
    {
        std::cout << "No joint values provided." << std::endl;
        return;
    }

    // Prepare cuda memory
    int num_of_joints = joint_values[0].size();
    int num_of_links = link_maps.size();
    int num_of_config = joint_values.size();
    int num_of_collision_spheres = collision_spheres_map.size();
    int joint_values_size = num_of_config * num_of_joints;
    int joint_values_bytes = joint_values_size * sizeof(float);
    int joint_types_bytes = joint_types.size() * sizeof(int);
    int size_of_pose_matrix = 4 * 4 * sizeof(float); // We do not need the last row of the matrix
    int joint_poses_bytes = joint_poses.size() * size_of_pose_matrix;
    int joint_axes_bytes = joint_axes.size() * sizeof(float) * 3;
    int link_maps_bytes = link_maps.size() * sizeof(int);
    int link_poses_set_size = num_of_links * num_of_config * size_of_pose_matrix;
    int link_poses_set_bytes = link_poses_set_size * sizeof(float);
    int collision_spheres_map_bytes = num_of_collision_spheres * sizeof(int);
    int collision_spheres_pos_bytes = num_of_collision_spheres * sizeof(float) * 3;
    int collision_spheres_pos_in_baselink_size = num_of_collision_spheres * num_of_config * 3;
    int collision_spheres_pos_in_baselink_bytes = collision_spheres_pos_in_baselink_size * sizeof(float);

    //******************* */
    // std::cout << "input check" << std::endl;
    // for (int i = 0; i < num_of_collision_spheres; i++)
    // {
    //     std::cout << "cs[" << i << "]: " << collision_spheres_pos[i][0] << " " << collision_spheres_pos[i][1] << " " << collision_spheres_pos[i][2] << " " << collision_spheres_map[i] << std::endl;
    // }
    // ******************* */


    // Allocate device memory
    float *d_joint_values;
    int *d_joint_types;
    float *d_joint_poses;
    float *d_joint_axes;
    int *d_link_maps;
    int *d_collision_spheres_map;
    float *d_collision_spheres_pos;
    float *d_link_poses_set;
    float *d_collision_spheres_pos_in_baselink;

    cudaMalloc(&d_joint_values, joint_values_bytes);
    cudaMalloc(&d_joint_types, joint_types_bytes);
    cudaMalloc(&d_joint_poses, joint_poses_bytes);
    cudaMalloc(&d_joint_axes, joint_axes_bytes);
    cudaMalloc(&d_link_maps, link_maps_bytes);
    cudaMalloc(&d_collision_spheres_map, collision_spheres_map_bytes);
    cudaMalloc(&d_collision_spheres_pos, collision_spheres_pos_bytes);
    cudaMalloc(&d_link_poses_set, link_poses_set_bytes);
    cudaMalloc(&d_collision_spheres_pos_in_baselink, collision_spheres_pos_in_baselink_bytes);

    // Copy data from host to device
    cudaMemcpy(d_joint_values, floatVectorFlatten(joint_values).data(), joint_values_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_joint_types, joint_types.data(), joint_types_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_joint_poses, IsometryVectorFlatten(joint_poses).data(), joint_poses_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_joint_axes, Vector3dflatten(joint_axes).data(), joint_axes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_link_maps, link_maps.data(), link_maps_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_collision_spheres_map, collision_spheres_map.data(), collision_spheres_map_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_collision_spheres_pos, floatVectorFlatten(collision_spheres_pos).data(), collision_spheres_pos_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_of_config + threadsPerBlock - 1) / threadsPerBlock;

    kin_forward_collision_spheres_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_joint_values, 
        num_of_joints,
        num_of_config,
        d_joint_types,
        d_joint_poses,
        num_of_links,
        d_joint_axes,
        d_link_maps,
        num_of_collision_spheres,
        d_collision_spheres_map,
        d_collision_spheres_pos,
        d_link_poses_set,
        d_collision_spheres_pos_in_baselink
    );
    cudaDeviceSynchronize();

    std::vector<float> h_link_poses_set(link_poses_set_size);
    std::vector<float> h_collision_spheres_pos_in_baselink(collision_spheres_pos_in_baselink_size);
    cudaMemcpy(h_link_poses_set.data(), d_link_poses_set, link_poses_set_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_collision_spheres_pos_in_baselink.data(), d_collision_spheres_pos_in_baselink, collision_spheres_pos_in_baselink_bytes, cudaMemcpyDeviceToHost);

    link_poses_set.clear();
    collision_spheres_pos_in_baselink.clear();
    for (int i = 0; i < num_of_config; i++)
    {
        link_poses_set.push_back(fromFloatVectorToIsometry3d(std::vector<float>(h_link_poses_set.begin() + i * num_of_links * 16, h_link_poses_set.begin() + (i + 1) * num_of_links * 16)));
        std::vector<std::vector<float>> collision_spheres_pos_in_baselink_of_current_config;
        for ( int j = 0; j < num_of_collision_spheres; j++)
        {
            collision_spheres_pos_in_baselink_of_current_config.push_back(std::vector<float>(h_collision_spheres_pos_in_baselink.begin() + i * num_of_collision_spheres * 3 + j * 3, h_collision_spheres_pos_in_baselink.begin() + i * num_of_collision_spheres * 3 + (j + 1) * 3));
        }
        collision_spheres_pos_in_baselink.push_back(collision_spheres_pos_in_baselink_of_current_config);
    }

    // Free device memory
    cudaFree(d_joint_values);
    cudaFree(d_joint_types);
    cudaFree(d_joint_poses);
    cudaFree(d_joint_axes);
    cudaFree(d_link_maps);
    cudaFree(d_collision_spheres_map);
    cudaFree(d_collision_spheres_pos);
    cudaFree(d_link_poses_set);
    cudaFree(d_collision_spheres_pos_in_baselink);
}

void CUDAMPLib::evaluation_cuda(
    const std::vector<std::vector<float>>& joint_values,
    const std::vector<int>& joint_types,
    const std::vector<Eigen::Isometry3d>& joint_poses,
    const std::vector<Eigen::Vector3d>& joint_axes,
    const std::vector<int>& link_maps,
    const std::vector<int>& collision_spheres_map,
    const std::vector<std::vector<float>>& collision_spheres_pos,
    const std::vector<float>& collision_spheres_radius,
    const std::vector<CostBasePtr>& costs,
    std::vector<float>& costs_values,
    std::vector<std::vector<std::vector<float>>>& collision_spheres_pos_in_baselink_for_debug
)
{
    if (joint_values.size() == 0)
    {
        std::cout << "No joint values provided." << std::endl;
        return;
    }

    // Prepare cuda memory
    int num_of_joints = joint_values[0].size();
    int num_of_links = link_maps.size();
    int num_of_config = joint_values.size();
    int num_of_collision_spheres = collision_spheres_map.size();
    int joint_values_size = num_of_config * num_of_joints;
    int joint_values_bytes = joint_values_size * sizeof(float);
    int joint_types_bytes = joint_types.size() * sizeof(int);
    int size_of_pose_matrix = 4 * 4 * sizeof(float); // We do not need the last row of the matrix
    int joint_poses_bytes = joint_poses.size() * size_of_pose_matrix;
    int joint_axes_bytes = joint_axes.size() * sizeof(float) * 3;
    int link_maps_bytes = link_maps.size() * sizeof(int);
    int link_poses_set_size = num_of_links * num_of_config * size_of_pose_matrix;
    int link_poses_set_bytes = link_poses_set_size * sizeof(float);
    int collision_spheres_map_bytes = num_of_collision_spheres * sizeof(int);
    int collision_spheres_pos_bytes = num_of_collision_spheres * sizeof(float) * 3;
    int collision_spheres_radius_bytes = num_of_collision_spheres * sizeof(float);
    int collision_spheres_pos_in_baselink_size = num_of_collision_spheres * num_of_config * 3;
    int collision_spheres_pos_in_baselink_bytes = collision_spheres_pos_in_baselink_size * sizeof(float);
    int num_of_costs = costs.size();
    int single_cost_bytes = num_of_config * sizeof(float);
    int cost_bytes = single_cost_bytes * num_of_costs; // for each configuration and each cost, we have a cost value
    

    // Allocate device memory
    float *d_joint_values;
    int *d_joint_types;
    float *d_joint_poses;
    float *d_joint_axes;
    int *d_link_maps;
    int *d_collision_spheres_map;
    float *d_collision_spheres_pos;
    float *d_collision_spheres_radius;
    float *d_link_poses_set;
    float *d_collision_spheres_pos_in_baselink;
    float *d_cost;

    cudaMalloc(&d_joint_values, joint_values_bytes);
    cudaMalloc(&d_joint_types, joint_types_bytes);
    cudaMalloc(&d_joint_poses, joint_poses_bytes);
    cudaMalloc(&d_joint_axes, joint_axes_bytes);
    cudaMalloc(&d_link_maps, link_maps_bytes);
    cudaMalloc(&d_collision_spheres_map, collision_spheres_map_bytes);
    cudaMalloc(&d_collision_spheres_pos, collision_spheres_pos_bytes);
    cudaMalloc(&d_collision_spheres_radius, collision_spheres_radius_bytes);
    cudaMalloc(&d_link_poses_set, link_poses_set_bytes);
    cudaMalloc(&d_collision_spheres_pos_in_baselink, collision_spheres_pos_in_baselink_bytes);
    cudaMalloc(&d_cost, cost_bytes);

    // Copy data from host to device
    cudaMemcpy(d_joint_values, floatVectorFlatten(joint_values).data(), joint_values_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_joint_types, joint_types.data(), joint_types_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_joint_poses, IsometryVectorFlatten(joint_poses).data(), joint_poses_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_joint_axes, Vector3dflatten(joint_axes).data(), joint_axes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_link_maps, link_maps.data(), link_maps_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_collision_spheres_map, collision_spheres_map.data(), collision_spheres_map_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_collision_spheres_pos, floatVectorFlatten(collision_spheres_pos).data(), collision_spheres_pos_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_collision_spheres_radius, collision_spheres_radius.data(), collision_spheres_radius_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_of_config + threadsPerBlock - 1) / threadsPerBlock;

    kin_forward_collision_spheres_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_joint_values, 
        num_of_joints,
        num_of_config,
        d_joint_types,
        d_joint_poses,
        num_of_links,
        d_joint_axes,
        d_link_maps,
        num_of_collision_spheres,
        d_collision_spheres_map,
        d_collision_spheres_pos,
        d_link_poses_set,
        d_collision_spheres_pos_in_baselink
    );
    cudaDeviceSynchronize();

    // ready the self collision spheres as output for debug
    std::vector<float> h_collision_spheres_pos_in_baselink(collision_spheres_pos_in_baselink_size);
    cudaMemcpy(h_collision_spheres_pos_in_baselink.data(), d_collision_spheres_pos_in_baselink, collision_spheres_pos_in_baselink_bytes, cudaMemcpyDeviceToHost);

    collision_spheres_pos_in_baselink_for_debug.clear();
    for (int i = 0; i < num_of_config; i++)
    {
        std::vector<std::vector<float>> collision_spheres_pos_in_baselink_of_current_config;
        for ( int j = 0; j < num_of_collision_spheres; j++)
        {
            collision_spheres_pos_in_baselink_of_current_config.push_back(std::vector<float>(h_collision_spheres_pos_in_baselink.begin() + i * num_of_collision_spheres * 3 + j * 3, h_collision_spheres_pos_in_baselink.begin() + i * num_of_collision_spheres * 3 + (j + 1) * 3));
        }
        collision_spheres_pos_in_baselink_for_debug.push_back(collision_spheres_pos_in_baselink_of_current_config);
    }

    for (size_t i = 0; i < costs.size(); i++)
    {
        float* d_current_cost = &d_cost[i * num_of_config];
        costs[i]->computeCost(
            d_joint_values, // joint values
            num_of_config, // number of configurations
            d_collision_spheres_pos_in_baselink, // robot collision spheres position in base link frame
            d_collision_spheres_radius, // robot collision spheres radius
            num_of_collision_spheres, // number of robot collision spheres
            d_current_cost);
    }

    // Copy the cost values from device to host
    costs_values.resize(num_of_config);
    for (size_t i = 0; i < costs.size(); i++)
    {
        std::vector<float> current_cost_values(num_of_config);
        float* d_current_cost = &d_cost[i * num_of_config];
        cudaMemcpy(current_cost_values.data(), d_current_cost, single_cost_bytes, cudaMemcpyDeviceToHost);
        for (int j = 0; j < num_of_config; j++)
        {
            costs_values[j] += current_cost_values[j];
        }
    }

    // Free device memory
    cudaFree(d_joint_values);
    cudaFree(d_joint_types);
    cudaFree(d_joint_poses);
    cudaFree(d_joint_axes);
    cudaFree(d_link_maps);
    cudaFree(d_collision_spheres_map);
    cudaFree(d_collision_spheres_pos);
    cudaFree(d_collision_spheres_radius);
    cudaFree(d_link_poses_set);
    cudaFree(d_collision_spheres_pos_in_baselink);
    cudaFree(d_cost);
}
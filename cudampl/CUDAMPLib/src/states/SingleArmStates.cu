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
                default:
                    printf("Unknown joint type: %d\n", joint_types[i]);
                    break;
            }
        }
    }

    // __global__ void update_collision_spheres_kernel(
    //     const int num_of_states,
    //     const int num_of_links,
    //     const int num_of_self_collision_spheres,
    //     const int* __restrict__ collision_spheres_map,
    //     const float* __restrict__ collision_spheres_pos, // collision sphere position in link frame
    //     const float* __restrict__ link_poses_set,
    //     float* collision_spheres_pos_in_baselink
    // )
    // {
    //     int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //     if (idx >= num_of_states * num_of_self_collision_spheres)
    //         return;

    //     int state_idx = idx / num_of_self_collision_spheres;
    //     int sphere_idx = idx % num_of_self_collision_spheres;

    //     const float* collision_sphere_pos = &collision_spheres_pos[sphere_idx * 3]; // collision sphere position in link frame
    //     float* collision_sphere_pos_in_baselink = &collision_spheres_pos_in_baselink[state_idx * num_of_self_collision_spheres * 3 + sphere_idx * 3]; // collision sphere position in base link frame
    //     const float* link_pose = &link_poses_set[state_idx * num_of_links * 16 + collision_spheres_map[sphere_idx] * 16]; // link pose in base link frame

    //     collision_sphere_pos_in_baselink[0] = link_pose[0] * collision_sphere_pos[0] + link_pose[1] * collision_sphere_pos[1] + link_pose[2] * collision_sphere_pos[2] + link_pose[3];
    //     collision_sphere_pos_in_baselink[1] = link_pose[4] * collision_sphere_pos[0] + link_pose[5] * collision_sphere_pos[1] + link_pose[6] * collision_sphere_pos[2] + link_pose[7];
    //     collision_sphere_pos_in_baselink[2] = link_pose[8] * collision_sphere_pos[0] + link_pose[9] * collision_sphere_pos[1] + link_pose[10] * collision_sphere_pos[2] + link_pose[11];
    // }

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
            if (d_active_joint_map[i] == 1) {
                float diff = d_states_1[state_1_idx * num_of_joints + i] - d_states_2[state_2_idx * num_of_joints + i];
                sum += diff * diff;
            }
        }

        d_distances[idx] = sqrtf(sum);
    }

    __global__ void filterStatesKernel(
        const int* d_filter,              // 0/1 flag per state
        const int* d_prefix,              // exclusive prefix sum of d_filter
        const float* d_joint_states,      // original joint states
        float* d_joint_states_new,        // output joint states
        const float* d_link_poses,        // original link poses
        float* d_link_poses_new,          // output link poses
        const float* d_spheres,           // original self-collision spheres
        float* d_spheres_new,             // output self-collision sphere positions
        int initial_num_of_states,        // total number of states before filtering
        int num_joints,                   // number of joint values per state
        int link_elements,                // number of floats per state for link poses (num_of_links * 4 * 4)
        int sphere_elements               // number of floats per state for spheres (num_of_self_collision_spheres * 3)
    )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < initial_num_of_states) {
            if (d_filter[i]) {  // if state i is feasible
                int new_index = d_prefix[i];
                // Copy joint states (each state has num_joints floats)
                for (int j = 0; j < num_joints; j++) {
                    d_joint_states_new[new_index * num_joints + j] =
                        d_joint_states[i * num_joints + j];
                }
                // Copy link poses (each state has link_elements floats)
                for (int j = 0; j < link_elements; j++) {
                    d_link_poses_new[new_index * link_elements + j] =
                        d_link_poses[i * link_elements + j];
                }
                // Copy self-collision sphere data (each state has sphere_elements floats)
                for (int j = 0; j < sphere_elements; j++) {
                    d_spheres_new[new_index * sphere_elements + j] =
                        d_spheres[i * sphere_elements + j];
                }
            }
        }
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
        size_t d_self_collision_spheres_pos_in_base_link_bytes = (size_t)num_of_states * space_info->num_of_self_collision_spheres * 3 * sizeof(float);

        auto allocate_result = cudaMalloc(&d_joint_states, d_joint_states_bytes);
        if (allocate_result != cudaSuccess) {
            cudaGetLastError();
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
            cudaGetLastError();
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
            cudaGetLastError();
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            std::cerr << "Free memory: " << free_memory / (1024 * 1024) << " MB" << std::endl;
            std::cerr << "Requested d_self_collision_spheres_pos_in_base_link_bytes: " << d_self_collision_spheres_pos_in_base_link_bytes / (1024 * 1024) << " MB" << std::endl;
            // print in red
            std::cerr << "\033[31m" << "CUDA Error: " << cudaGetErrorString(allocate_result) << " d_self_collision_spheres_pos_in_base_link_bytes is too large " << "\033[0m" << std::endl;

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
            cudaFree(d_self_collision_spheres_pos_in_base_link);
            // set the pointers to nullptr for safety
            d_joint_states = nullptr;
            d_link_poses_in_base_link = nullptr;
            d_self_collision_spheres_pos_in_base_link = nullptr;
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
            cudaFree(d_self_collision_spheres_pos_in_base_link);
        }
        else{
            // static_cast the space_info to SingleArmSpaceInfo
            SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

            float * d_joint_states_new;
            float * d_link_poses_in_base_link_new;
            float * d_self_collision_spheres_pos_in_base_link_new;

            size_t d_joint_states_new_bytes = new_num_of_states * this->num_of_joints * sizeof(float);
            size_t d_link_poses_in_base_link_new_bytes = new_num_of_states * single_arm_space_info->num_of_links * 4 * 4 * sizeof(float);
            size_t d_self_collision_spheres_pos_in_base_link_new_bytes = new_num_of_states * single_arm_space_info->num_of_self_collision_spheres * 3 * sizeof(float);

            // Allocate memory for the joint states
            cudaMalloc(&d_joint_states_new, d_joint_states_new_bytes);
            cudaMalloc(&d_link_poses_in_base_link_new, d_link_poses_in_base_link_new_bytes);
            cudaMalloc(&d_self_collision_spheres_pos_in_base_link_new, d_self_collision_spheres_pos_in_base_link_new_bytes);

            // Copy the joint states from the old memory to the new memory
            int j = 0;
            for (int i = 0; i < initial_num_of_states; i++)
            {
                if (filter_map[i])
                {
                    // copy asynchrounously
                    cudaMemcpyAsync(d_joint_states_new + j * num_of_joints, d_joint_states + i * num_of_joints, num_of_joints * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(d_link_poses_in_base_link_new + j * single_arm_space_info->num_of_links * 4 * 4, d_link_poses_in_base_link + i * single_arm_space_info->num_of_links * 4 * 4, single_arm_space_info->num_of_links * 4 * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(d_self_collision_spheres_pos_in_base_link_new + j * single_arm_space_info->num_of_self_collision_spheres * 3, d_self_collision_spheres_pos_in_base_link + i * single_arm_space_info->num_of_self_collision_spheres * 3, single_arm_space_info->num_of_self_collision_spheres * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
                    j++;
                }
            }

            // Wait for the copy to finish
            cudaDeviceSynchronize();

            // Free the old memory
            cudaFree(d_joint_states);
            cudaFree(d_link_poses_in_base_link);
            cudaFree(d_self_collision_spheres_pos_in_base_link);

            // Update the pointers
            d_joint_states = d_joint_states_new;
            d_link_poses_in_base_link = d_link_poses_in_base_link_new;
            d_self_collision_spheres_pos_in_base_link = d_self_collision_spheres_pos_in_base_link_new;
        }

        CUDA_CHECK(cudaGetLastError());
    }

    void SingleArmStates::cudaFilterStates(const std::vector<bool>& filter_map)
    {
        int initial_num_of_states = num_of_states_;

        // Call the base class cudaFilterStates to update num_of_states_
        BaseStates::cudaFilterStates(filter_map);
        size_t new_num_of_states = num_of_states_;

        if (new_num_of_states == 0) {
            cudaFree(d_joint_states);
            cudaFree(d_link_poses_in_base_link);
            cudaFree(d_self_collision_spheres_pos_in_base_link);
            return;
        }

        // Cast the space_info to SingleArmSpaceInfo
        SingleArmSpaceInfoPtr single_arm_space_info =
            std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

        int num_joints = this->num_of_joints;
        int num_links = single_arm_space_info->num_of_links;
        int num_spheres = single_arm_space_info->num_of_self_collision_spheres;
        
        // Each link pose is a 4x4 matrix (16 floats)
        int link_elements = num_links * 16;
        // Each self-collision sphere is represented by 3 floats
        int sphere_elements = num_spheres * 3;

        // Allocate new device memory for the filtered data
        float* d_joint_states_new = nullptr;
        float* d_link_poses_new = nullptr;
        float* d_spheres_new = nullptr;
        size_t d_joint_states_new_bytes = new_num_of_states * num_joints * sizeof(float);
        size_t d_link_poses_new_bytes = new_num_of_states * link_elements * sizeof(float);
        size_t d_spheres_new_bytes = new_num_of_states * sphere_elements * sizeof(float);
        cudaMalloc(&d_joint_states_new, d_joint_states_new_bytes);
        cudaMalloc(&d_link_poses_new, d_link_poses_new_bytes);
        cudaMalloc(&d_spheres_new, d_spheres_new_bytes);

        // Create a device vector for the filter (convert bools to ints: 0 or 1)
        thrust::device_vector<int> d_filter(initial_num_of_states);
        for (int i = 0; i < initial_num_of_states; i++) {
            d_filter[i] = filter_map[i] ? 1 : 0;
        }

        // Compute the exclusive prefix sum to determine new indices
        thrust::device_vector<int> d_prefix(initial_num_of_states);
        thrust::exclusive_scan(d_filter.begin(), d_filter.end(), d_prefix.begin());

        // Launch the kernel to compact the states in parallel.
        int threadsPerBlock = 256;
        int blocks = (initial_num_of_states + threadsPerBlock - 1) / threadsPerBlock;
        filterStatesKernel<<<blocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_filter.data()),
            thrust::raw_pointer_cast(d_prefix.data()),
            d_joint_states,
            d_joint_states_new,
            d_link_poses_in_base_link,
            d_link_poses_new,
            d_self_collision_spheres_pos_in_base_link,
            d_spheres_new,
            initial_num_of_states,
            num_joints,
            link_elements,
            sphere_elements
        );
        cudaDeviceSynchronize(); // Ensure the kernel has completed

        // Free old device memory
        cudaFree(d_joint_states);
        cudaFree(d_link_poses_in_base_link);
        cudaFree(d_self_collision_spheres_pos_in_base_link);

        // Update the pointers to point to the new filtered data
        d_joint_states = d_joint_states_new;
        d_link_poses_in_base_link = d_link_poses_new;
        d_self_collision_spheres_pos_in_base_link = d_spheres_new;
    }

    std::vector<std::vector<float>> SingleArmStates::getJointStatesHost() const
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

        // update the self collision spheres position in base link frame
        blocksPerGrid = (num_of_states_ * space_info_single_arm_space->num_of_self_collision_spheres + threadsPerBlock - 1) / threadsPerBlock;
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

    void SingleArmStates::print() const
    {
        // static_cast the space_info to SingleArmSpaceInfo
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);

        // Get the joint states
        std::vector<std::vector<float>> joint_states = getJointStatesHost();

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

        // static cast the query states to SingleArmStates
        SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info_);

        // static cast the states to SingleArmStates
        SingleArmStatesPtr single_arm_states = std::static_pointer_cast<SingleArmStates>(query_states);
        float * d_query_joint_states = single_arm_states->getJointStatesCuda();

        neighbors_index.clear();

        if (k > num_of_states_){
            // set k to num_of_states
            k = num_of_states_;
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
            // find index of the k least distances of distances_from_query_to_states[i]
            std::vector<int> index_k_nearest_neighbors = kLeastIndices(distances_from_query_to_states[i], k);
            neighbors_index.push_back(index_k_nearest_neighbors);
        }

        // free the memory
        cudaFree(d_distances_from_query_to_states);

        return k;
    }

    int SingleArmStateManager::find_k_nearest_neighbors(
        int k, const BaseStatesPtr & query_states, 
        std::vector<std::vector<int>> & neighbors_index,
        const std::vector<std::vector<int>> & group_indexs
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
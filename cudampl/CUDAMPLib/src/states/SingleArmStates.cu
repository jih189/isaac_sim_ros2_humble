#include <states/SingleArmStates.h>

namespace CUDAMPLib
{
    /**
        * @brief Multiply two 4x4 matrices
     */
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

    /**
        * @brief Set the matrix to identity
     */
    __device__ void set_identity(float* M)
    {
        M[0]  = 1.f;  M[1]  = 0.f;  M[2]  = 0.f;  M[3]  = 0.f;
        M[4]  = 0.f;  M[5]  = 1.f;  M[6]  = 0.f;  M[7]  = 0.f;
        M[8]  = 0.f;  M[9]  = 0.f;  M[10] = 1.f;  M[11] = 0.f;
        M[12] = 0.f;  M[13] = 0.f;  M[14] = 0.f;  M[15] = 1.f;
    }

    /**
        * @brief Forward kinematics for a fixed joint
     */
    __device__ void fixed_joint_fn_cuda(
        float* parent_link_pose,
        float* joint_pose,
        float* link_pose
    )
    {
        multiply4x4(parent_link_pose, joint_pose, link_pose);
    }

    /**
        * @brief Get the rotation matrix from axis-angle representation
     */
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

    /**
        * @brief Generate the link pose for a revolute joint
     */
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

    /**
        * @brief Generate the link pose for a prismatic joint
     */
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
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    link_poses_set[idx * num_of_links * 16 + i * 4 + j] = 0.0f;
                }
                link_poses_set[idx * num_of_links * 16 + i * 4 + i] = 1.0f;
            }

            // Calculate forward kinematics for each link
            // size_t j = 0;
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
    
    SingleArmStates::SingleArmStates(int num_of_states, SingleArmSpaceInfoPtr space_info, int num_of_joints)
    : BaseStates(num_of_states, space_info)
    {
        this->num_of_joints = num_of_joints;

        // Allocate memory for the joint states
        cudaMalloc(&d_joint_states, num_of_states * num_of_joints * sizeof(float));
        cudaMalloc(&d_link_poses_in_base_link, num_of_states * space_info->num_of_links * 4 * 4 * sizeof(float));
        cudaMalloc(&d_self_collision_spheres_pos_in_base_link, num_of_states * space_info->num_of_self_collision_spheres * 3 * sizeof(float));
    }

    SingleArmStates::~SingleArmStates()
    {
        // Free the memory
        cudaFree(d_joint_states);
        cudaFree(d_link_poses_in_base_link);
        cudaFree(d_self_collision_spheres_pos_in_base_link);
    }

    std::vector<std::vector<float>> SingleArmStates::getJointStatesHost()
    {
        // Allocate memory for the joint states
        std::vector<float> joint_states_flatten(num_of_states * num_of_joints, 0.0);

        // Copy the joint states from device to host
        cudaMemcpy(joint_states_flatten.data(), d_joint_states, num_of_states * num_of_joints * sizeof(float), cudaMemcpyDeviceToHost);

        // Reshape the joint states
        std::vector<std::vector<float>> joint_states(num_of_states, std::vector<float>(num_of_joints, 0.0));
        for (int i = 0; i < num_of_states; i++)
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
        std::vector<float> self_collision_spheres_pos_in_base_link_flatten(num_of_states * space_info_single_arm_space->num_of_self_collision_spheres * 3, 0.0);

        // Copy the self collision spheres position in base link frame from device to host
        cudaMemcpy(self_collision_spheres_pos_in_base_link_flatten.data(), d_self_collision_spheres_pos_in_base_link, num_of_states * space_info_single_arm_space->num_of_self_collision_spheres * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // Reshape the self collision spheres position in base link frame
        std::vector<std::vector<std::vector<float>>> self_collision_spheres_pos_in_base_link(num_of_states, std::vector<std::vector<float>>(space_info_single_arm_space->num_of_self_collision_spheres, std::vector<float>(3, 0.0)));

        for (int i = 0; i < num_of_states; i++)
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

    void SingleArmStates::update()
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_of_states + threadsPerBlock - 1) / threadsPerBlock;
        SingleArmSpaceInfoPtr space_info_single_arm_space = std::static_pointer_cast<SingleArmSpaceInfo>(this->space_info);
        
        // Update the states
        kin_forward_collision_spheres_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_joint_states,
            num_of_joints,
            num_of_states,
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
} // namespace CUDAMPLib
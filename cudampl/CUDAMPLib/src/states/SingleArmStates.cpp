#include <states/SingleArmStates.h>

namespace CUDAMPLib
{
    SingleArmStates::SingleArmStates(int num_of_states, SingleArmSpaceInfoPtr space_info, int num_of_joints)
    : BaseStates(num_of_states, space_info)
    {
        this->num_of_joints = num_of_joints;

        // Allocate memory for the joint states
        cudaMalloc(&d_joint_states, num_of_states * num_of_joints * sizeof(float));
    }

    SingleArmStates::~SingleArmStates()
    {
        // Free the memory
        cudaFree(d_joint_states);
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
} // namespace CUDAMPLib
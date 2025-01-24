#include <states/SingleArmStates.h>

namespace CUDAMPLib
{
    SingleArmStates::SingleArmStates(int num_of_states, int num_of_joints)
    : BaseStates(num_of_states)
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
} // namespace CUDAMPLib
#include <motions/SingleArmMotions.h>

namespace CUDAMPLib
{
    SingleArmMotions::SingleArmMotions(int num_of_motions, SingleArmSpaceInfoPtr space_info)
    : BaseMotions(num_of_motions, space_info)
    {
        this->num_of_joints = space_info->num_of_joints;

        // Allocate memory for the motions
        cudaMalloc(&d_joint_states_1, num_of_motions * this->num_of_joints * sizeof(float));
        cudaMalloc(&d_joint_states_2, num_of_motions * this->num_of_joints * sizeof(float));
    }

    SingleArmMotions::~SingleArmMotions() {
        // Free the memory
        if (num_of_motions > 0) {
            cudaFree(d_joint_states_1);
            cudaFree(d_joint_states_2);
        }
    }
} // namespace CUDAMPLib
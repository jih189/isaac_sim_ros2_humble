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

    void SingleArmMotions::print() const {
        // Copy the joint states to the host
        float *h_joint_states_1 = new float[num_of_motions * num_of_joints];
        float *h_joint_states_2 = new float[num_of_motions * num_of_joints];

        cudaMemcpy(h_joint_states_1, d_joint_states_1, num_of_motions * num_of_joints * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_joint_states_2, d_joint_states_2, num_of_motions * num_of_joints * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Number of motions: %d\n", num_of_motions);
        // print each pair of joint states
        for (int i = 0; i < num_of_motions; i++) {
            printf("Motion %d: \n", i);
            for (int j = 0; j < num_of_joints; j++) {
                printf("%f ", h_joint_states_1[i * num_of_joints + j]);
            }
            printf("\n");
            for (int j = 0; j < num_of_joints; j++) {
                printf("%f ", h_joint_states_2[i * num_of_joints + j]);
            }
            printf("\n");
        }
    }
} // namespace CUDAMPLib
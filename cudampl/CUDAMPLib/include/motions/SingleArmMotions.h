#pragma once

#include <base/Motions.h>
#include <states/SingleArmStates.h>

namespace CUDAMPLib
{
    class SingleArmMotions : public BaseMotions
    {
        public:
            SingleArmMotions(int num_of_motions, SingleArmSpaceInfoPtr space_info);
            ~SingleArmMotions();

            float * getJointStates1Cuda() {
                return d_joint_states_1;
            }

            float * getJointStates2Cuda() {
                return d_joint_states_2;
            }

            std::vector<std::vector<float>> getJointStates1Host() const;
            std::vector<std::vector<float>> getJointStates2Host() const;

            void print() const override;

        private:
            float * d_joint_states_1; // joint states of the first end of the motion
            float * d_joint_states_2; // joint states of the second end of the motion
            int num_of_joints;
    };

    typedef std::shared_ptr<SingleArmMotions> SingleArmMotionsPtr;

} // namespace CUDAMPLib
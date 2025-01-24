#pragma once

#include <base/States.h>
#include <vector>
#include <stdexcept>


namespace CUDAMPLib
{
    class SingleArmStates : public BaseStates
    {
        public:
            SingleArmStates(int num_of_states, int num_of_joints);
            
            ~SingleArmStates();

            float * getJointStatesCuda() {
                return d_joint_states;
            }

        private:
            int num_of_joints;
            float * d_joint_states; // joint states of each state
    };

    typedef std::shared_ptr<SingleArmStates> SingleArmStatesPtr;
} // namespace CUDAMPLibs
#include <base/States.h>
#include <vector>
#include <stdexcept>


namespace CUDAMPLib
{
    class SingleArmStates : public BaseStates
    {
        public:

            SingleArmStates(const std::vector<std::vector<float>> & joint_values);
            
            ~SingleArmStates() override;

            void setJointStates(float * d_joint_states) {
                this->d_joint_states = d_joint_states;
            }

            float * getJointStates() {
                return d_joint_states;
            }

        private:
            int num_of_joints;
            float * d_joint_states; // joint states of each state
    };
} // namespace CUDAMPLibs
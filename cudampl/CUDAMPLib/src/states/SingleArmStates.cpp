#include <states/SingleArmStates.h>

namespace CUDAMPLib
{
    SingleArmStates::SingleArmStates(const std::vector<std::vector<float>> & joint_values)
    {
        // Given a set of joint values, we need to allocate cuda memory for them.
        num_of_states = joint_values.size();

        // If there are no joint values, raise an error
        if (num_of_states == 0)
        {
            throw std::runtime_error("No joint values provided.");
        }

        num_of_joints = joint_values[0].size();

    }

    SingleArmStates::~SingleArmStates()
    {
        // Cleanup code here, if needed
        
    }
} // namespace CUDAMPLib
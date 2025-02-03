#include <tasks/SingleArmTask.h>

namespace CUDAMPLib
{
    SingleArmTask::SingleArmTask(const std::vector<std::vector<float>>& start_joint_values, const std::vector<std::vector<float>>& goal_joint_values)
    {
        // Do something
    }

    SingleArmTask::~SingleArmTask()
    {
        // Do something
    }

    BaseStatesPtr SingleArmTask::getStartStates()
    {
        return start_states;
    }

    BaseStatesPtr SingleArmTask::getGoalStates()
    {
        return goal_states;
    }
} // namespace CUDAMPLib
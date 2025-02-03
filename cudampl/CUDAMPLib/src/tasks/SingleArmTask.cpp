#include <tasks/SingleArmTask.h>

namespace CUDAMPLib
{
    SingleArmTask::SingleArmTask(const std::vector<std::vector<float>>& start_joint_values, const std::vector<std::vector<float>>& goal_joint_values)
    {
        // copy the start and goal joint values
        start_states_vector = start_joint_values;
        goal_states_vector = goal_joint_values;
    }

    SingleArmTask::~SingleArmTask()
    {
        // Do something
    }

    std::vector<std::vector<float>> SingleArmTask::getStartStatesVector()
    {
        return start_states_vector;
    }

    std::vector<std::vector<float>> SingleArmTask::getGoalStatesVector()
    {
        return goal_states_vector;
    }
} // namespace CUDAMPLib
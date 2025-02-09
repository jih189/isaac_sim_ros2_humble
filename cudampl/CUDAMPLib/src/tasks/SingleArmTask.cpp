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

    BaseStatesPtr SingleArmTask::getStartStates(BaseSpacePtr space)
    {
        // statics cast the space to SingleArmSpace
        auto single_arm_space = std::static_pointer_cast<SingleArmSpace>(space);
        // Create the start states from the start states vector
        return single_arm_space->createStatesFromVector(start_states_vector);
    }

    BaseStatesPtr SingleArmTask::getGoalStates(BaseSpacePtr space)
    {
        // statics cast the space to SingleArmSpace
        auto single_arm_space = std::static_pointer_cast<SingleArmSpace>(space);
        // Create the goal states from the goal states vector
        return single_arm_space->createStatesFromVector(goal_states_vector);
    }

    std::vector<std::vector<float>> SingleArmTask::getStartStatesVector()
    {
        return start_states_vector;
    }

    std::vector<std::vector<float>> SingleArmTask::getGoalStatesVector()
    {
        return goal_states_vector;
    }

    void SingleArmTask::setSolution(const BaseStatesPtr& solution, const BaseSpacePtr space)
    {
        has_solution = true;

        // statics cast the space to SingleArmSpace
        auto single_arm_space = std::static_pointer_cast<SingleArmSpace>(space);

        // get the joint vector in active joints from the solution
        solution_vector = single_arm_space->getJointVectorInActiveJointsFromStates(solution);
    }

    std::vector<std::vector<float>> SingleArmTask::getSolution()
    {
        return solution_vector;
    }
} // namespace CUDAMPLib
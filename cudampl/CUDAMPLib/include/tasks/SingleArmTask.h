#pragma once

#include "base/Task.h"

namespace CUDAMPLib
{
    class SingleArmTask : public BaseTask
    {
        public:
            SingleArmTask(const std::vector<std::vector<float>>& start_joint_values, const std::vector<std::vector<float>>& goal_joint_values);
            ~SingleArmTask();
            std::vector<std::vector<float>> getStartStatesVector();
            std::vector<std::vector<float>> getGoalStatesVector();
        private:
            std::vector<std::vector<float>> start_states_vector;
            std::vector<std::vector<float>> goal_states_vector;
    };

    typedef std::shared_ptr<SingleArmTask> SingleArmTaskPtr;
} // namespace CUDAMPLibs
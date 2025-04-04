#pragma once

#include "base/Task.h"

namespace CUDAMPLib
{
    class SingleArmTask : public BaseTask
    {
        public:
            SingleArmTask(const std::vector<std::vector<float>>& start_joint_values, const std::vector<std::vector<float>>& goal_joint_values);
            SingleArmTask(const std::vector<std::vector<float>>& start_joint_values, const BaseSpacePtr& goal_region, int goal_config_sampling_attempts=50);
            ~SingleArmTask();
            std::vector<std::vector<float>> getStartStatesVector();
            std::vector<std::vector<float>> getGoalStatesVector();
            BaseStatesPtr getStartStates(BaseSpacePtr space) override;
            BaseStatesPtr getGoalStates(BaseSpacePtr space) override;
            void setSolution(const BaseStatesPtr& solution, BaseSpacePtr space) override;
            std::vector<std::vector<float>> getSolution();
        private:
            std::vector<std::vector<float>> start_states_vector;
            std::vector<std::vector<float>> goal_states_vector;
            std::vector<std::vector<float>> solution_vector;
    };

    typedef std::shared_ptr<SingleArmTask> SingleArmTaskPtr;
} // namespace CUDAMPLibs
#pragma once

#include "base/Task.h"
#include "states/SingleArmStates.h"

namespace CUDAMPLib
{
    class SingleArmTask : public BaseTask
    {
        public:
            SingleArmTask(const std::vector<std::vector<float>>& start_joint_values, const std::vector<std::vector<float>>& goal_joint_values);
            ~SingleArmTask();
            BaseStatesPtr getStartStates() override;
            BaseStatesPtr getGoalStates() override;
        private:
            SingleArmStatesPtr start_states;
            SingleArmStatesPtr goal_states;
    };
} // namespace CUDAMPLibs
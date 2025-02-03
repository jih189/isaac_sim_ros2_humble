#pragma once

#include <memory>
#include <base/States.h>

namespace CUDAMPLib
{
    class BaseTask{
        public:
            virtual ~BaseTask() {}

            /**
                * @brief Get the start states.
                * @return The start states.
             */
            virtual BaseStatesPtr getStartStates() = 0;
            /**
                * @brief Get the goal states.
                * @return The goal states.
             */
            virtual BaseStatesPtr getGoalStates() = 0;
    };

    typedef std::shared_ptr<BaseTask> BaseTaskPtr;
} // namespace CUDAMPLibs
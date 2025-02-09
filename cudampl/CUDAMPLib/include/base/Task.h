#pragma once

#include <memory>
#include <base/Space.h>
#include <spaces/SingleArmSpace.h>

namespace CUDAMPLib
{
    class BaseTask{
        public:
            BaseTask() : has_solution(false) {}
            virtual ~BaseTask() {}
            virtual BaseStatesPtr getStartStates(BaseSpacePtr space) = 0;
            virtual BaseStatesPtr getGoalStates(BaseSpacePtr space) = 0;
            virtual void setSolution(const BaseStatesPtr& solution, const BaseSpacePtr space) = 0;
            bool hasSolution() const { return has_solution; }

        protected:
            bool has_solution;
    };

    typedef std::shared_ptr<BaseTask> BaseTaskPtr;
} // namespace CUDAMPLibs
#pragma once

#include <memory>
#include <base/Space.h>
#include <spaces/SingleArmSpace.h>

namespace CUDAMPLib
{
    class BaseTask{
        public:
            virtual ~BaseTask() {}
            virtual BaseStatesPtr getStartStates(BaseSpacePtr space) = 0;
            virtual BaseStatesPtr getGoalStates(BaseSpacePtr space) = 0;
    };

    typedef std::shared_ptr<BaseTask> BaseTaskPtr;
} // namespace CUDAMPLibs
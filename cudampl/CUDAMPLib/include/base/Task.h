#pragma once

#include <memory>
#include <base/Space.h>
#include <spaces/SingleArmSpace.h>

namespace CUDAMPLib
{
    class BaseTask{
        public:
            BaseTask() : has_solution(false) {
                failure_reason = "";
            }
            virtual ~BaseTask() {}
            virtual BaseStatesPtr getStartStates(BaseSpacePtr space) = 0;
            virtual BaseStatesPtr getGoalStates(BaseSpacePtr space) = 0;
            virtual void setSolution(const BaseStatesPtr& solution, const BaseSpacePtr space) = 0;
            bool hasSolution() const { return has_solution; }
            void setFailureReason(const std::string& reason) { failure_reason = reason; }
            std::string getFailureReason() const { return failure_reason; }

        protected:
            bool has_solution;
            std::string failure_reason;
    };

    typedef std::shared_ptr<BaseTask> BaseTaskPtr;
} // namespace CUDAMPLibs
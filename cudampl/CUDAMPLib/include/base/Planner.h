#pragma once

#include <base/Task.h>
#include <base/Space.h>
#include <base/Termination.h>
#include <stdexcept>

namespace CUDAMPLib
{
    class BasePlanner
    {
        public:
            // constructor
            BasePlanner(BaseSpacePtr space) {space_ = space;}
            virtual ~BasePlanner() {}
            virtual void setMotionTask(BaseTaskPtr task, bool get_full_path) = 0;
            virtual void solve(BaseTerminationPtr termination_condition) = 0;
            
            /** get planning space */
            BaseSpacePtr getSpace() {
                return space_;
            }
        protected:
            BaseSpacePtr space_;
            bool get_full_path_;
    };

    typedef std::shared_ptr<BasePlanner> BasePlannerPtr;
} // namespace CUDAMPLibs
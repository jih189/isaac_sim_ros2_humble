#pragma once

#include <base/Task.h>
#include <base/Space.h>
#include <stdexcept>

namespace CUDAMPLib
{
    class BasePlanner
    {
        public:
            // constructor
            BasePlanner(BaseSpacePtr space) {space_ = space;}
            virtual ~BasePlanner() {}
            virtual void setMotionTask(BaseTaskPtr task) = 0;
            virtual void solve() = 0;
            
            /** get planning space */
            BaseSpacePtr getSpace() {
                return space_;
            }
        protected:
            BaseSpacePtr space_;
    };

    typedef std::shared_ptr<BasePlanner> BasePlannerPtr;
} // namespace CUDAMPLibs
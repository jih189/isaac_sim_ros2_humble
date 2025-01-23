#pragma once

#include <base/Task.h>
#include <base/Space.h>
#include <stdexcept>

namespace CUDAMPLib
{
    class BasePlanner
    {
        public:
            virtual ~BasePlanner() {}
            virtual void setMotionTask(BaseTaskPtr task) = 0;
            virtual void solve() = 0;
            
            /** check if space is set */
            bool hasSpace() { return has_space_; }
            /** get planning space */
            BaseSpacePtr getSpace() {
                if (!has_space_)
                    throw std::runtime_error("Planner has no space");
                return space_;
            }
            /** set planning space */
            void setSpace(BaseSpacePtr space) { space_ = space; has_space_ = true; }


        private:
            BaseSpacePtr space_;
            bool has_space_ = false;
    };

    typedef std::shared_ptr<BasePlanner> BasePlannerPtr;
} // namespace CUDAMPLibs
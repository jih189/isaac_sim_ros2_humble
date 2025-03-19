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
            /**
                @brief Set the motion task.
                @param task The motion task.
                @param get_full_path If true, the planner will return the full path. Otherwise, it will return the waypoints.
             */
            virtual void setMotionTask(BaseTaskPtr task, bool get_full_path) = 0;

            /**
                @brief Solve the motion task.
                @param termination_condition The termination condition.
             */
            virtual void solve(BaseTerminationPtr termination_condition) = 0;
            
            /**
                @brief get planning space. 
             */
            BaseSpacePtr getSpace() {
                return space_;
            }
        protected:
            BaseSpacePtr space_;
            bool get_full_path_;
    };

    typedef std::shared_ptr<BasePlanner> BasePlannerPtr;
} // namespace CUDAMPLibs
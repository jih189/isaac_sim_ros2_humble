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

            /**
                @brief Get the start states.
             */
            virtual BaseStatesPtr getStartStates(BaseSpacePtr space) = 0;

            /**
                @brief Get the goal states.
             */
            virtual BaseStatesPtr getGoalStates(BaseSpacePtr space) = 0;

            /**
                @brief Get the start states vector.
             */
            virtual std::vector<std::vector<float>> getStartStatesVector() = 0;

            /**
                @brief Get the goal states vector.
             */
            virtual std::vector<std::vector<float>> getGoalStatesVector() = 0;

            /**
                @brief Set the solution.
                @param solution The solution.
                @param space The space.
             */
            virtual void setSolution(const BaseStatesPtr& solution, const BaseSpacePtr space) = 0;

            /**
                @brief Set the solution with a vector.
             */
            virtual void setSolution(const std::vector<std::vector<float>>& solution) = 0;

            /**
                @brief Check if the task has a solution.
             */
            bool hasSolution() const { return has_solution; }
            
            /**
                @brief Set the has solution flag and set the failure reason.
                @param reason The reason of failure.
             */
            void setFailureReason(const std::string& reason) { failure_reason = reason; }

            /**
                @brief Get the failure reason.
                @return The reason of failure.
             */
            std::string getFailureReason() const { return failure_reason; }

        protected:
            bool has_solution;
            std::string failure_reason;
    };

    typedef std::shared_ptr<BaseTask> BaseTaskPtr;
} // namespace CUDAMPLibs
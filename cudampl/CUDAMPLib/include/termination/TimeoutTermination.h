#pragma once

#include <base/Termination.h>
#include <chrono>
#include <cstdio>

namespace CUDAMPLib
{
    class TimeoutTermination : public BaseTermination
    {
        public:
            TimeoutTermination(float timeout) : timeout_(timeout) {}
            void reset() override;
            bool checkTerminationCondition() override;
            void printTerminationReason() override
            {
                // print in yellow color
                printf("\033[1;33m Timeout reached \033[0m \n");
            }
        private:
            float timeout_;
            std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    };
    typedef std::shared_ptr<TimeoutTermination> TimeoutTerminationPtr;

} // namespace CUDAMPLib
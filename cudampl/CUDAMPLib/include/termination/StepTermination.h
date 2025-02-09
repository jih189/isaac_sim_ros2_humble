#pragma once

#include <base/Termination.h>
#include <cstdio>

namespace CUDAMPLib
{
    class StepTermination : public BaseTermination
    {
        public:
            StepTermination(int step_limit) : step_limit_(step_limit) {}
            void reset() override;
            bool checkTerminationCondition() override;
            void printTerminationReason() override
            {
                // print in yellow color
                printf("\033[1;33m Step limit reached \033[0m \n");
            }
        private:
            int step_limit_;
            int step_count_;
    };
    typedef std::shared_ptr<StepTermination> StepTerminationPtr;

} // namespace CUDAMPLib
#include <termination/StepTermination.h>
namespace CUDAMPLib
{
    void StepTermination::reset()
    {
        step_count_ = 0;
    }

    bool StepTermination::checkTerminationCondition()
    {
        step_count_++;
        return step_count_ >= step_limit_;
    }
} // namespace CUDAMPLib
#include <termination/TimeoutTermination.h>

namespace CUDAMPLib
{
    void TimeoutTermination::reset()
    {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    bool TimeoutTermination::checkTerminationCondition()
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time_).count();
        return elapsed_time >= timeout_;
    }
} // namespace CUDAMPLib
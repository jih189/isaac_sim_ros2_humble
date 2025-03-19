#pragma once

#include <memory>

namespace CUDAMPLib
{
    class BaseTermination
    {
        public:
            virtual ~BaseTermination() {}

            /**
                @brief reset the termination condition.
             */
            virtual void reset() = 0;

            /**
                @brief check if the termination condition is met and return true if it is met.
             */
            virtual bool checkTerminationCondition() = 0;

            /**
                @brief print the termination reason.
             */
            virtual void printTerminationReason() = 0;
    };
    typedef std::shared_ptr<BaseTermination> BaseTerminationPtr;
} // namespace CUDAMPLib
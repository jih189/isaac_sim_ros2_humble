#pragma once

#include <base/States.h>
#include <base/Motions.h>

namespace CUDAMPLib
{
    class BaseConstraint {
        public:
            // virtual destructor. We need to define how to clean the cuda memory in the derived class.
            virtual ~BaseConstraint() {}
            virtual void computeCost(BaseStatesPtr states) = 0;
            virtual void computeCost(BaseMotionsPtr motions) = 0;
    };

    typedef std::shared_ptr<BaseConstraint> BaseConstraintPtr;
} // namespace CUDAMPLibs
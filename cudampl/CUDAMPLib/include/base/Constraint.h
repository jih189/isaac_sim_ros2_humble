#pragma once

#include <base/States.h>
#include <base/Motions.h>

namespace CUDAMPLib
{
    class BaseConstraint {
        public:
            BaseConstraint(std::string name) : constraint_name(name) {}
            // virtual destructor. We need to define how to clean the cuda memory in the derived class.
            virtual ~BaseConstraint() {}
            
            virtual void computeCost(BaseStatesPtr states) = 0;
            virtual void computeCost(BaseMotionsPtr motions) = 0;

            std::string getName() const { return constraint_name; }

        protected:
            std::string constraint_name;
    };

    typedef std::shared_ptr<BaseConstraint> BaseConstraintPtr;
} // namespace CUDAMPLibs
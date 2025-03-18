#pragma once

#include <base/States.h>

namespace CUDAMPLib
{
    #define CUDAMPLib_PROJECT_MAX_ITERATION 10 // maximum iteration for projection. This number should not be too large.
    
    class BaseConstraint {
        public:
            BaseConstraint(std::string name, bool is_projectable) : constraint_name(name), is_projectable_(is_projectable) {}
            // virtual destructor. We need to define how to clean the cuda memory in the derived class.
            virtual ~BaseConstraint() {}
            
            /**
                @brief Compute the cost of the states based on the constraint.
             */
            virtual void computeCost(BaseStatesPtr states) = 0;

            bool isProjectable() { return is_projectable_; }

            virtual void computeGradientAndError(BaseStatesPtr states) 
            {
                if (!is_projectable_)
                {
                    throw std::runtime_error("This constraint is not projectable.");
                }
                else{
                    throw std::runtime_error("The function computeGradientAndError is not implemented.");
                }
            }

            std::string getName() const { return constraint_name; }

            int getConstraintIndex(const SpaceInfoPtr space_info) {
                for (int i = 0; i < space_info->num_of_constraints; i++)
                    if (space_info->constraint_names[i] == constraint_name)
                        return i;
                return -1;
            }

        protected:
            std::string constraint_name;
            bool is_projectable_;
    };

    typedef std::shared_ptr<BaseConstraint> BaseConstraintPtr;
} // namespace CUDAMPLibs
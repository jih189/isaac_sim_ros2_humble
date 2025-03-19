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
                @brief Compute the cost of the states and save the cost value into the states.
             */
            virtual void computeCost(BaseStatesPtr states) = 0;

            /**
                @brief Check if the constraint is projectable.
             */
            bool isProjectable() { return is_projectable_; }

            /**
                @brief Compute the gradient and error of the states and save the gradient and error into the states.
             */
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

            /**
                @brief Get the name of the constraint.
             */
            std::string getName() const { return constraint_name; }

            /**
                @brief Get the constraint index in the space info.
                @param space_info The space info
                @return The index of the constraint in the space info. If not found, return -1.
             */
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
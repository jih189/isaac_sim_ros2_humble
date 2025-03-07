#pragma once

#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include <base/Constraint.h>
#include <states/SingleArmStates.h>
#include <util.h>
#include <cuda_runtime.h>

namespace CUDAMPLib
{
    class BoundaryConstraint : public BaseConstraint
    {
        public:
            /**
                * @brief Construct a new Boundary Constraint object
                * @param constraint_name The name of the constraint.
                * @param lower_bound The lower bound of the states.
                * @param upper_bound The upper bound of the states.
                * @param active_joint_map The active joint map.
             */
            BoundaryConstraint(
                const std::string& constraint_name,
                const std::vector<float>& lower_bound,
                const std::vector<float>& upper_bound,
                const std::vector<bool>& active_joint_map
            );
            ~BoundaryConstraint() override;

            void computeCost(BaseStatesPtr states) override;

            void computeCostLarge(BaseStatesPtr states) override;

            void computeCostFast(BaseStatesPtr states) override;

            void computeGradientAndError(BaseStatesPtr states) override;

        private:
            float * d_lower_bound_full_;
            float * d_upper_bound_full_;
            int * d_active_joint_map_as_int_;
    };

    typedef std::shared_ptr<BoundaryConstraint> BoundaryConstraintPtr;
} // namespace CUDAMPLib

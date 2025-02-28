#pragma once

#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include <base/Constraint.h>
#include <states/SingleArmStates.h>
#include <util.h>
#include <cuda_runtime.h>

namespace CUDAMPLib
{
    class SelfCollisionConstraint : public BaseConstraint
    {
        public:
            /**
                * @brief Construct a new Self Collision Constraint object
                * @param self_collision_enables_map
             */
            SelfCollisionConstraint(
                const std::string& constraint_name,
                const std::vector<std::vector<bool>>& self_collision_enables_map
            );
            ~SelfCollisionConstraint() override;

            void computeCost(BaseStatesPtr states) override;

            void computeCostLarge(BaseStatesPtr states) override;

            void computeCostFast(BaseStatesPtr states) override;
        
        private:
            int *d_self_collision_enables_map;
    };

    typedef std::shared_ptr<SelfCollisionConstraint> SelfCollisionConstraintPtr;
} // namespace CUDAMPLibs
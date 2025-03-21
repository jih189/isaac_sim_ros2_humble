#pragma once

#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include <base/Constraint.h>
#include <states/SingleArmStates.h>
#include <util.h>
#include <cuda_runtime.h>

namespace CUDAMPLib
{
    class EnvConstraintSphere : public BaseConstraint
    {
        public:
            /**
                @brief Construct a new Env Constraint object
                @param env_collision_spheres_pos The positions of the environment collision spheres in the base_link frame.
                @param env_collision_spheres_radius The radii of the environment collision spheres.
             */
            EnvConstraintSphere(
                const std::string& constraint_name,
                const std::vector<std::vector<float>>& env_collision_spheres_pos,
                const std::vector<float>& env_collision_spheres_radius
            );
            ~EnvConstraintSphere() override;

            void computeCost(BaseStatesPtr states) override;
            
        private:
            int num_of_env_collision_spheres;
            float *d_env_collision_spheres_pos_in_base_link;
            float *d_env_collision_spheres_radius;
    };

    typedef std::shared_ptr<EnvConstraintSphere> EnvConstraintSpherePtr;
} // namespace CUDAMPLibs
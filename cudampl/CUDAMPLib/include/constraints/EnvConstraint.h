#pragma once

#include <base/Constraint.h>
#include <util.h>
#include <vector>

namespace CUDAMPLib
{
    class EnvConstraint : public BaseConstraint
    {
        public:
            /**
                * @brief Construct a new Env Constraint object
                * @param env_collision_spheres_pos The positions of the environment collision spheres in the base_link frame.
                * @param env_collision_spheres_radius The radii of the environment collision spheres.
             */
            EnvConstraint(
                const std::string& constraint_name,
                const std::vector<std::vector<float>>& env_collision_spheres_pos,
                const std::vector<float>& env_collision_spheres_radius
            );
            ~EnvConstraint() override;

            void computeCost(BaseStatesPtr states, float* output) override;
            void computeCost(BaseMotionsPtr motions, float* output) override;
        
        private:
            int num_of_env_collision_spheres;
            float *d_env_collision_spheres_pos_in_base_link;
            float *d_env_collision_spheres_radius;
    };

    typedef std::shared_ptr<EnvConstraint> EnvConstraintPtr;
} // namespace CUDAMPLibs
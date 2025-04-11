#pragma once

#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include <base/Constraint.h>
#include <states/SingleArmStates.h>
#include <util.h>
#include <cuda_runtime.h>

namespace CUDAMPLib
{
    class EnvConstraintCylinder : public BaseConstraint
    {
        public:
            /**
                @brief Construct a new Env Constraint object
                @param env_collision_cylinders_pos The positions of the environment collision cylinders in the base_link frame.
                @param env_collision_cylinders_orientation The orientations of the environment collision cylinders in the base_link frame.
                @param env_collision_cylinders_radius The radii of the environment collision cylinders.
                @param env_collision_cylinders_height The heights of the environment collision cylinders.
             */
            EnvConstraintCylinder(
                const std::string& constraint_name,
                const std::vector<std::vector<float>>& env_collision_cylinders_pos,
                const std::vector<std::vector<float>>& env_collision_cylinders_orientation,
                const std::vector<float>& env_collision_cylinders_radius,
                const std::vector<float>& env_collision_cylinders_height
            );
            ~EnvConstraintCylinder() override;

            void computeCost(BaseStatesPtr states) override;

            std::string generateCheckConstraintCode() override;

            std::string generateLaunchCheckConstraintCode() override;

            
        private:
            int num_of_env_collision_cylinders;
            float *d_env_collision_cylinders_inverse_pose_matrix_in_base_link;
            float *d_env_collision_cylinders_radius;
            float *d_env_collision_cylinders_height;
    };

    typedef std::shared_ptr<EnvConstraintCylinder> EnvConstraintCylinderPtr;
} // namespace CUDAMPLibs
#pragma once

#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include <base/Constraint.h>
#include <states/SingleArmStates.h>
#include <util.h>
#include <cuda_runtime.h>

namespace CUDAMPLib
{
    class EnvConstraintCuboid : public BaseConstraint
    {
        public:
            /**
                @brief Construct a new Env Constraint object
                @param env_collision_cuboid_pos The positions of the environment collision cuboids in the base_link frame.
                @param env_collision_cuboid_orientation The orientations of the environment collision cuboids in the base_link frame.
                @param env_collision_cuboid_max The maximum extents of the environment collision cuboids.
                @param env_collision_cuboid_min The minimum extents of the environment collision cuboids.
             */
            EnvConstraintCuboid(
                const std::string& constraint_name,
                const std::vector<std::vector<float>>& env_collision_cuboid_pos,
                const std::vector<std::vector<float>>& env_collision_cuboid_orientation,
                const std::vector<std::vector<float>>& env_collision_cuboid_max,
                const std::vector<std::vector<float>>& env_collision_cuboid_min
            );
            ~EnvConstraintCuboid() override;

            void computeCost(BaseStatesPtr states) override;

            std::string generateCheckConstraintCode() override;

            std::string generateLaunchCheckConstraintCode() override;

            
        private:
            int num_of_env_collision_cuboids;
            float *d_env_collision_cuboids_inverse_pose_matrix_in_base_link;
            float *d_env_collision_cuboids_max;
            float *d_env_collision_cuboids_min;
    };

    typedef std::shared_ptr<EnvConstraintCuboid> EnvConstraintCuboidPtr;
} // namespace CUDAMPLibs
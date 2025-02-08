#pragma once

#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20014

#include <stdexcept>

#include <base/Space.h>
#include <states/SingleArmStates.h>
#include <graphs/SingleArmGraph.h>
#include <vector>
#include <util.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <random>
#include <ctime>
#include <limits>

namespace CUDAMPLib
{
    class SingleArmSpace : public BaseSpace
    {
        public:
            SingleArmSpace(
                size_t dim,
                const std::vector<BaseConstraintPtr>& constraints,
                const std::vector<int>& joint_types,
                const std::vector<Eigen::Isometry3d>& joint_poses,
                const std::vector<Eigen::Vector3d>& joint_axes,
                const std::vector<int>& link_parent_link_maps,
                const std::vector<int>& collision_spheres_to_link_map,
                const std::vector<std::vector<float>>& collision_spheres_pos_in_link,
                const std::vector<float>& collision_spheres_radius,
                const std::vector<bool>& active_joint_map,
                const std::vector<float>& lower,
                const std::vector<float>& upper,
                const std::vector<float>& default_joint_values
            );

            ~SingleArmSpace() override;

            const std::vector<float>& getLower() const { return lower_bound; }

            const std::vector<float>& getUpper() const { return upper_bound; }

            BaseStatesPtr sample(int num_of_config) override;

            void checkMotions(
                const BaseMotionsPtr & motions,
                std::vector<bool>& motion_feasibility,
                std::vector<float>& motion_costs
            ) override;

            void checkMotions(
                const BaseStatesPtr & states1, 
                const BaseStatesPtr & states2, 
                std::vector<bool>& motion_feasibility,
                std::vector<float>& motion_costs
            ) override;

            void checkStates(
                const BaseStatesPtr & states, 
                std::vector<bool>& state_feasibility
            ) override;

            void checkStates(const BaseStatesPtr & states) override;

            BaseGraphPtr createGraph() override;

            BaseStateManagerPtr createStateManager() override; 

            void getSpaceInfo(SingleArmSpaceInfoPtr space_info);

            /**
                @brief helper function to create a set of states based on task.
                This is used later to create start states and goal states.
                @param joint_values The joint values of the states. Be careful with the size of the joint values.
                                    The size of the joint values should be equal to the number of active joints not the number of joints.
                @return The states.
             */
            BaseStatesPtr createStatesFromVector(const std::vector<std::vector<float>>& joint_values);


            /**
                @brief helper function to create a set of states based on task.
                This is used for motion checking.
                @param joint_values The joint values of the states include non-active joints.
                @return The states.
             */
            BaseStatesPtr createStatesFromVectorFull(const std::vector<std::vector<float>>& joint_values);

        private:

            int num_of_joints; // number of joints where joints include fixed joints.
            int num_of_links;
            int num_of_self_collision_spheres;
            std::vector<float> lower_bound;
            std::vector<float> upper_bound;
            std::vector<bool> active_joint_map_;
            std::vector<float> default_joint_values_;

            // variables for gpu memory
            int * d_joint_types;
            float * d_joint_poses;
            float * d_joint_axes;
            int * d_link_parent_link_maps;
            int * d_collision_spheres_to_link_map;
            float * d_self_collision_spheres_pos_in_link;
            float * d_self_collision_spheres_radius;
            int * d_active_joint_map;
            float * d_lower_bound;
            float * d_upper_bound;
            float * d_default_joint_values;

            // random number generator
            // std::random_device rd;  // Non-deterministic seed (preferred)
            std::mt19937 gen;       // Standard mersenne_twister_engine seeded with rd()
            std::uniform_int_distribution<unsigned long> dist;
    };

    typedef std::shared_ptr<SingleArmSpace> SingleArmSpacePtr;
} // namespace CUDAMPLibs
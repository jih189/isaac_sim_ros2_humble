#pragma once

#include <base/Space.h>
#include <vector>
#include <util.h>

namespace CUDAMPLib
{
    class SingleArmSpace : public BaseSpace
    {
        public:
            SingleArmSpace(
                size_t dim,
                const std::vector<int>& joint_types,
                const std::vector<Eigen::Isometry3d>& joint_poses,
                const std::vector<Eigen::Vector3d>& joint_axes,
                const std::vector<int>& link_parent_link_maps,
                const std::vector<int>& collision_spheres_to_link_map,
                const std::vector<std::vector<float>>& collision_spheres_pos_in_link,
                const std::vector<float>& collision_spheres_radius
            );

            ~SingleArmSpace() override;

            void sample(int num_of_config, std::vector<std::vector<float>>& samples) override;

            void getMotions(
                const std::vector<std::vector<float>>& start, 
                const std::vector<std::vector<float>>& end, 
                std::vector<std::vector<std::vector<float>>>& motions,
                std::vector<bool> motion_feasibility
            ) override;

            void checkMotions(
                const std::vector<std::vector<float>>& start, 
                const std::vector<std::vector<float>>& end, 
                std::vector<bool>& motion_feasibility
            ) override;

            void checkStates(
                const std::vector<std::vector<float>>& states, 
                std::vector<bool>& state_feasibility
            ) override;

        private:
            int num_of_joints; // number of joints where joints include fixed joints.
            int num_of_links;
            int num_of_self_collision_spheres;

            // variables for gpu memory
            int * d_joint_types;
            float * d_joint_poses;
            float * d_joint_axes;
            int * d_link_parent_link_maps;
            int * d_collision_spheres_to_link_map;
            float * d_collision_spheres_pos_in_link;
            float * d_collision_spheres_radius;
    };

    typedef std::shared_ptr<SingleArmSpace> SingleArmSpacePtr;
} // namespace CUDAMPLibs
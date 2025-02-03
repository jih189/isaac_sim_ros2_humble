#pragma once

#include <base/States.h>
#include <vector>
#include <stdexcept>


namespace CUDAMPLib
{
    #define CUDAMPLib_UNKNOWN 0
    #define CUDAMPLib_REVOLUTE 1
    #define CUDAMPLib_PRISMATIC 2
    #define CUDAMPLib_PLANAR 3
    #define CUDAMPLib_FLOATING 4
    #define CUDAMPLib_FIXED 5

    struct SingleArmSpaceInfo : public SpaceInfo
    {
        // The information of the space
        int num_of_joints;
        int num_of_links;
        int num_of_self_collision_spheres;
        std::vector<float> lower_bound;
        std::vector<float> upper_bound;

        // address of the device memory about the space information
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
    };
    typedef std::shared_ptr<SingleArmSpaceInfo> SingleArmSpaceInfoPtr;

    class SingleArmStates : public BaseStates
    {
        public:
            SingleArmStates(int num_of_states, SingleArmSpaceInfoPtr space_info);
            
            ~SingleArmStates();

            /**
                @brief Get the joint states in device memory.
                @return The joint states in device memory.
             */
            float * getJointStatesCuda() {
                return d_joint_states;
            }

            /**
                @brief Get the link poses in base link in device memory.
             */
            float * getLinkPosesInBaseLinkCuda() {
                return d_link_poses_in_base_link;
            }

            /**
                @brief Get the self collision spheres in base link in device memory.
             */
            float * getSelfCollisionSpheresPosInBaseLinkCuda() {
                return d_self_collision_spheres_pos_in_base_link;
            }

            /**
                @brief Get the self collision spheres in base link in host memory.
             */
            std::vector<std::vector<std::vector<float>>> getSelfCollisionSpheresPosInBaseLinkHost();

            /**
                * @brief Get the joint states in host memory.
                * @return The joint states in host memory. The joint states are represented as a 
                          list of configurations, each represented as a list of floats.
             */
            std::vector<std::vector<float>> getJointStatesHost();
            

            /**
                * Based on the current states, update link poses, joint poses, and joint axes, collision spheres in base link, etc.
             */
            void update() override;

        private:
            int num_of_joints; // number of joints. This includes fixed joints.
            float * d_joint_states; // joint states of each state
            float * d_link_poses_in_base_link; // link poses in base link
            float * d_self_collision_spheres_pos_in_base_link; // collision spheres in base link
    };

    typedef std::shared_ptr<SingleArmStates> SingleArmStatesPtr;
} // namespace CUDAMPLibs
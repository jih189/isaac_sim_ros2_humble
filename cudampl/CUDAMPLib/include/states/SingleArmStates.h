#pragma once

#include <base/States.h>
#include <vector>
#include <stdexcept>


namespace CUDAMPLib
{
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
        float * d_collision_spheres_pos_in_link;
        float * d_collision_spheres_radius;
        int * d_active_joint_map;
        float * d_lower_bound;
        float * d_upper_bound;
        float * d_default_joint_values;
    };
    typedef std::shared_ptr<SingleArmSpaceInfo> SingleArmSpaceInfoPtr;

    class SingleArmStates : public BaseStates
    {
        public:
            SingleArmStates(int num_of_states, SingleArmSpaceInfoPtr space_info, int num_of_joints);
            
            ~SingleArmStates();

            /**
                * @brief Get the joint states in device memory.
                * @return The joint states in device memory.
             */
            float * getJointStatesCuda() {
                return d_joint_states;
            }

            /**
                * @brief Get the joint states in host memory.
                * @return The joint states in host memory. The joint states are represented as a 
                          list of configurations, each represented as a list of floats.
             */
            std::vector<std::vector<float>> getJointStatesHost();

        private:
            int num_of_joints; // number of joints. This includes fixed joints.
            float * d_joint_states; // joint states of each state
    };

    typedef std::shared_ptr<SingleArmStates> SingleArmStatesPtr;
} // namespace CUDAMPLibs
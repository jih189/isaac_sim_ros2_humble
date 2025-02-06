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
                @brief Filter the states based on the filter_map
                @param filter_map The filter map. If the value is true, the state is feasible. Otherwise, the state is infeasible.
             */
            void filterStates(const std::vector<bool> & filter_map) override;

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
            std::vector<std::vector<float>> getJointStatesHost() const;

            /**
                * Based on the current states, update link poses, joint poses, and joint axes, collision spheres in base link, etc.
             */
            void update() override;

            /**
                @brief Print the joint states. For debugging purposes.
             */
            void print() const override;

            int getNumOfJoints() const {
                return num_of_joints;
            }

        private:
            int num_of_joints; // number of joints. This includes fixed joints.
            float * d_joint_states; // joint states of each state
            float * d_link_poses_in_base_link; // link poses in base link
            float * d_self_collision_spheres_pos_in_base_link; // collision spheres in base link
    };
    typedef std::shared_ptr<SingleArmStates> SingleArmStatesPtr;

    template <typename T>
    class SingleArmKnn : public BaseKNearestNeighbors<T>{
        public:
            SingleArmKnn(SpaceInfoPtr space_info) : BaseKNearestNeighbors<T>(space_info) {
                // static cast the space_info to SingleArmSpaceInfoPtr
                SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(space_info);
                num_of_joints = single_arm_space_info->num_of_joints;
            }
            ~SingleArmKnn() {}

            void add_states(const BaseStatesPtr & states, const std::vector<T>& elems) override;

            std::vector<std::vector<T>> find_k_nearest_neighbors(int k, const BaseStatesPtr & query_states) override;

        private:
            int num_of_joints;
            float * d_joint_states;
    };
} // namespace CUDAMPLibs
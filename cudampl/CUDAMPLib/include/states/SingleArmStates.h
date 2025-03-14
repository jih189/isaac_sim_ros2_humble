#pragma once

#include <base/States.h>
#include <vector>
#include <stdexcept>
#include <util.h>

#include <NvrtcUtil.h>

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
        int num_of_active_joints;
        std::vector<float> lower_bound;
        std::vector<float> upper_bound;
        std::vector<std::string> link_names;
        std::vector<bool> active_joint_map;

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

        // kernel function of the space
        KernelFunctionPtr kernelFuncPtr;
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
                @brief Get the space jacobian in base link in device memory.
             */
            float * getSpaceJacobianInBaseLinkCuda() {
                return d_space_jacobian_in_base_link;
            }

            /**
                @brief Get the gradient of the states in device memory.
             */
            float * getGradientCuda() {
                return d_gradient;
            }

            float * getTotalGradientCuda() {
                return d_total_gradient;
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
                @brief Get the joint states in host memory.
                @return The joint states in host memory. The joint states are represented as a 
                          list of configurations, each represented as a list of floats.
             */
            std::vector<std::vector<float>> getJointStatesFullHost() const;

            /**
                @brief Get the joint states of active joints in host memory.
             */
            std::vector<std::vector<float>> getJointStatesHost() const;

            /**
                @brief Get link poses in base link in host memory.
                @return The link poses in base link in host memory. The link poses are represented as pose matrices.
            */
            std::vector<std::vector<Eigen::Isometry3d>> getLinkPosesInBaseLinkHost() const;

            /**
                @brief Get Link pose in base link in host memory of a specific link.
                @param link_name The name of the link
                @return The link pose in base link in host memory of a specific link.
             */
            std::vector<Eigen::Isometry3d> getLinkPoseInBaseLinkHost(std::string link_name) const;

            /**
                @brief Get the space jacobian in base link in host memory.
                @param link_name The name of the link
                @return The space jacobian in base link in host memory.
             */
            std::vector<Eigen::MatrixXd> getSpaceJacobianInBaseLinkHost(std::string link_name) const;

            /**
                * Based on the current states, update link poses, joint poses, and joint axes, collision spheres in base link, etc.
             */
            void update() override;

            void oldUpdate();

            /**
                @brief Calculate the forward kinematics of the states.
             */
            void calculateForwardKinematics();

            void calculateTotalGradientAndError(const std::vector<int> & constraint_indexs) override;

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
            float * d_space_jacobian_in_base_link; // jacobian in base link
            float * d_self_collision_spheres_pos_in_base_link; // collision spheres in base link
            float * d_gradient; // gradient of the states for each constraint
            float * d_total_gradient; // total gradient of the states
    };
    typedef std::shared_ptr<SingleArmStates> SingleArmStatesPtr;

    class SingleArmStateManager : public BaseStateManager{
        public:
            SingleArmStateManager(SpaceInfoPtr space_info) : BaseStateManager(space_info) {
                // static cast the space_info to SingleArmSpaceInfoPtr
                SingleArmSpaceInfoPtr single_arm_space_info = std::static_pointer_cast<SingleArmSpaceInfo>(space_info);
                num_of_joints = single_arm_space_info->num_of_joints;
                num_of_states_ = 0; // number of states in the knn
            }

            ~SingleArmStateManager();

            void clear() override;

            std::vector<int> add_states(const BaseStatesPtr & states) override;

            int find_k_nearest_neighbors(
                int k, const BaseStatesPtr & query_states, 
                std::vector<std::vector<int>> & neighbors_index
            ) override;

            int find_k_nearest_neighbors(
                int k, const BaseStatesPtr & query_states, 
                std::vector<std::vector<int>> & neighbors_index,
                const std::vector<std::vector<int>> & group_indexs
            ) override;

            BaseStatesPtr get_states(const std::vector<int> & states_index) override;

            BaseStatesPtr concatinate_states(const std::vector<BaseStatesPtr> & states) override;

        private:
            int num_of_joints;
            float * d_joint_states; // size is num_of_states_ * num_of_joints
    };
    typedef std::shared_ptr<SingleArmStateManager> SingleArmStateManagerPtr;
} // namespace CUDAMPLibs
#pragma once
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <Obstacle.h>

#include <NvrtcUtil.h>

namespace CPRRTC
{
    class RobotSolver
    {
        public:
            /**
                The Robot Solver will load the robot model and generate the cprrtc source code and use nvrtc to compile the kernel.
                After compiling the kernel, it will generate the ptx file as well and save it with the robot name.
                Later, if we want to use the same robot model, we can just load the ptx file and use it.
                @param robot_name The name of the robot.
                @param dim The dimension of the robot.
                @param joint_types The types of the joints. 1 for revolute, 2 for prismatic, 5 for fixed. We follow the Moveit convention.
                @param joint_poses The poses of the joints. The pose is in the link frame.
                @param joint_axes The axes of the joints. The axis is in the link frame.
                @param link_parent_link_maps The parent link map of the robot. 
                @param self_collision_spheres_to_link_map The self collision spheres to link map. The index of the sphere is the index of the link.
                @param self_collision_spheres_pos_in_link The position of the self collision spheres in the link frame.
                @param self_collision_spheres_radius The radius of the self collision spheres.
                @param active_joint_map The active joint map. The index of the joint is the index of the link.
                @param lower The lower bound of the joint limits.
                @param upper The upper bound of the joint limits.
                @param default_joint_values The default joint values.
                @param link_names The names of the links.
                @param resolution The resolution of the robot. The default value is 0.02.
            */
            RobotSolver(
                std::string robot_name,
                size_t dim,
                const std::vector<int>& joint_types,
                const std::vector<Eigen::Isometry3d>& joint_poses,
                const std::vector<Eigen::Vector3d>& joint_axes,
                const std::vector<int>& link_parent_link_maps,
                const std::vector<int>& self_collision_spheres_to_link_map,
                const std::vector<Eigen::Vector3d>& self_collision_spheres_pos_in_link,
                const std::vector<float>& self_collision_spheres_radius,
                const std::vector<bool>& active_joint_map,
                const std::vector<float>& lower,
                const std::vector<float>& upper,
                const std::vector<float>& default_joint_values,
                const std::vector<std::string>& link_names,
                float resolution = 0.02f
            );
            ~RobotSolver();

            /**
                @brief Set the environment obstacle cache. The cache will be used to check the collision.
                @param num_of_spheres The number of spheres in the environment.
                @param num_of_cuboids The number of cuboids in the environment.
                @param num_of_cylinders The number of cylinders in the environment.
            */
            void setEnvObstacleCache(int num_of_spheres, int num_of_cuboids, int num_of_cylinders);
            /**
                @brief Update the environment obstacle cache. The number of each obstacle should be less than the number of spheres, cuboids and cylinders respectively.
                @param spheres The spheres in the environment.
                @param cuboids The cuboids in the environment.
                @param cylinders The cylinders in the environment.
             */
            void updateEnvObstacle(std::vector<Sphere>& spheres, std::vector<Cuboid>& cuboids, std::vector<Cylinder>& cylinders);
            /**
                @brief Check the collision of the robot with the environment.
                @param joint_values The joint values of the robot. The size of the joint values should be equal to the number of active joints not the number of joints.
                @return Solution path. If the path is empty, it means it fails.
            */
            std::vector<std::vector<float>> solve(std::vector<float>& start, std::vector<float>& goal);

        private:
            std::string generateKernelSourceCode();
            std::string generateFKKernelSourceCode();

            // Add private member variables and methods here
            std::string robot_name_;
            size_t dim_;
            std::vector<int> joint_types_;
            std::vector<Eigen::Isometry3d> joint_poses_;
            std::vector<Eigen::Vector3d> joint_axes_;
            std::vector<int> link_parent_link_maps_;
            std::vector<int> self_collision_spheres_to_link_map_;
            std::vector<Eigen::Vector3d> self_collision_spheres_pos_in_link_;
            std::vector<float> self_collision_spheres_radius_;
            std::vector<bool> active_joint_map_;
            std::vector<float> lower_bound_;
            std::vector<float> upper_bound_;
            std::vector<float> default_joint_values_;
            std::vector<std::string> link_names_;
            float resolution_;
            int num_of_joints_;
            int num_of_links_;
            int num_of_self_collision_spheres_;
            int num_of_active_joints_;
            // self collision
            int num_of_self_collision_check_;
            std::vector<int> self_collision_sphere_indices_1_;
            std::vector<int> self_collision_sphere_indices_2_;
            std::vector<float> self_collision_distance_thresholds_;

            // Parameters
            int max_iterations_;
            int num_of_threads_per_motion_;
            int num_of_thread_blocks_;
            int max_step_;

    };

    // create std::shared_ptr for RobotSolver
    using RobotSolverPtr = std::shared_ptr<RobotSolver>;
} // namespace CUDAMPLib
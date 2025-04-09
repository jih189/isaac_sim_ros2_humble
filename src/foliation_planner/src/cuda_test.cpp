#include <rclcpp/rclcpp.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include "moveit/planning_interface/planning_interface.h"
#include "moveit/robot_state/conversions.h"
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit/robot_trajectory/robot_trajectory.h>

// cudampl include
#include <CUDAMPLib/spaces/SingleArmSpace.h>
#include <CUDAMPLib/constraints/EnvConstraintSphere.h>
#include <CUDAMPLib/constraints/EnvConstraintCuboid.h>
#include <CUDAMPLib/constraints/EnvConstraintCylinder.h>
#include <CUDAMPLib/constraints/SelfCollisionConstraint.h>
#include <CUDAMPLib/constraints/TaskSpaceConstraint.h>
#include <CUDAMPLib/constraints/BoundaryConstraint.h>
#include <CUDAMPLib/tasks/SingleArmTask.h>
#include <CUDAMPLib/planners/RRG.h>
#include <CUDAMPLib/planners/cRRTC.h>
#include <CUDAMPLib/termination/StepTermination.h>
#include <CUDAMPLib/termination/TimeoutTermination.h>

// ompl include
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>

#include "foliation_planner/robot_info.hpp"
#include "foliation_planner/obstacle_generator.hpp"
#include "foliation_planner/MPM_helper.hpp"

#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// include for time
#include <chrono>
#include <limits>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace ob = ompl::base;
namespace og = ompl::geometric;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("CUDAMPLib");

visualization_msgs::msg::MarkerArray generate_obstacles_markers(
    const std::vector<std::vector<float>> & balls_pos,
    const std::vector<float> & ball_radius,
    rclcpp::Node::SharedPtr node)
{
     // Create a obstacle MarkerArray message
    visualization_msgs::msg::MarkerArray obstacle_collision_spheres_marker_array;
    for (size_t i = 0; i < balls_pos.size(); i++)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "obstacle_collision_spheres";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = balls_pos[i][0];
        marker.pose.position.y = balls_pos[i][1];
        marker.pose.position.z = balls_pos[i][2];
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2 * ball_radius[i];
        marker.scale.y = 2 * ball_radius[i];
        marker.scale.z = 2 * ball_radius[i];
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.0;
        obstacle_collision_spheres_marker_array.markers.push_back(marker);
    }
    return obstacle_collision_spheres_marker_array;
}

std::vector<std::vector<double>> interpolate(const std::vector<double> & start, const std::vector<double> & goal, int num_points)
{
    std::vector<std::vector<double>> trajectory;
    for (int i = 0; i < num_points; ++i)
    {
        std::vector<double> point;
        for (size_t j = 0; j < start.size(); ++j)
        {
            double value = start[j] + (goal[j] - start[j]) * static_cast<double>(i) / static_cast<double>(num_points - 1);
            point.push_back(value);
        }
        trajectory.push_back(point);
    }
    return trajectory;
}

visualization_msgs::msg::MarkerArray generate_self_collision_markers(
    const std::vector<std::vector<float>> & collision_spheres_pos_of_selected_config,
    const std::vector<float> & collision_spheres_radius,
    rclcpp::Node::SharedPtr node)
{
    // Create a MarkerArray message
    visualization_msgs::msg::MarkerArray robot_collision_spheres_marker_array;
    for (size_t i = 0; i < collision_spheres_pos_of_selected_config.size(); i++)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = node->now();
        marker.ns = "self_collision_spheres";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = collision_spheres_pos_of_selected_config[i][0];
        marker.pose.position.y = collision_spheres_pos_of_selected_config[i][1];
        marker.pose.position.z = collision_spheres_pos_of_selected_config[i][2];
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2 * collision_spheres_radius[i];
        marker.scale.y = 2 * collision_spheres_radius[i];
        marker.scale.z = 2 * collision_spheres_radius[i];
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        robot_collision_spheres_marker_array.markers.push_back(marker);
    }
    return robot_collision_spheres_marker_array;
}

void generate_state_markers(
    const std::vector<std::vector<float>> & group_joint_values,
    const moveit::core::JointModelGroup* joint_model_group,
    moveit::core::RobotStatePtr robot_state,
    const std::string & group_ns,
    const std_msgs::msg::ColorRGBA color,
    visualization_msgs::msg::MarkerArray & robot_marker_array,
    std::vector<std::string> end_effector_link_names
)
{
    std::vector<visualization_msgs::msg::MarkerArray> group_state_markers;

    std::vector<std::string> display_links_names = joint_model_group->getLinkModelNames();

    // add end effector link names
    for (size_t i = 0; i < end_effector_link_names.size(); i++)
    {
        display_links_names.push_back(end_effector_link_names[i]);
    }

    for (size_t i = 0; i < group_joint_values.size(); i++)
    {
        std::vector<double> group_joint_values_i_double;
        for (size_t j = 0; j < group_joint_values[i].size(); j++)
        {
            group_joint_values_i_double.push_back((double)group_joint_values[i][j]);
        }

        robot_state->setJointGroupPositions(joint_model_group, group_joint_values_i_double);
        robot_state->update();
        visualization_msgs::msg::MarkerArray robot_marker;
        robot_state->getRobotMarkers(robot_marker, display_links_names, color, group_ns, rclcpp::Duration::from_seconds(0));
        group_state_markers.push_back(robot_marker);
    }

    robot_marker_array.markers.clear();

    // conbine group state markers
    for (size_t i = 0; i < group_state_markers.size(); i++)
    {
        robot_marker_array.markers.insert(robot_marker_array.markers.end(), group_state_markers[i].markers.begin(), group_state_markers[i].markers.end());
    }

    // update the id
    for (size_t i = 0; i < robot_marker_array.markers.size(); i++)
    {
        robot_marker_array.markers[i].id = i;
    }
}

void TEST_JACOBIAN(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    // create moveit robot state
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames()
    );

    std::string check_link_name = robot_info.getEndEffectorLinkName();

    // set a test joint values
    std::vector<float> joint_values_1 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> joint_values_2 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5};
    std::vector<float> joint_values_3 = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<std::vector<float>> joint_values_set;
    joint_values_set.push_back(joint_values_1);
    joint_values_set.push_back(joint_values_2);
    joint_values_set.push_back(joint_values_3);

    // create states based on the joint values
    auto states = single_arm_space->createStatesFromVector(joint_values_set);
    states->update();

    // statistic_cast_pointer_cast to SingleArmStates
    CUDAMPLib::SingleArmStatesPtr single_arm_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(states);

    std::vector<Eigen::Isometry3d> end_effector_link_poses_in_base_link = single_arm_states->getLinkPoseInBaseLinkHost(check_link_name);

    // print space jacobian
    std::vector<Eigen::MatrixXd> space_jacobian_in_base_link = single_arm_states->getSpaceJacobianInBaseLinkHost(check_link_name);

    for (size_t i = 0; i < end_effector_link_poses_in_base_link.size(); i++)
    {
        // print joint values
        std::cout << "Joint values: ";
        for (size_t j = 0; j < joint_values_set[i].size(); j++)
        {
            std::cout << joint_values_set[i][j] << " ";
        }
        std::cout << std::endl;
        std::cout << "End effector pose " << i << ": " << std::endl;
        std::cout << "position: " << end_effector_link_poses_in_base_link[i].translation().transpose() << std::endl;
        // std::cout << end_effector_link_poses_in_base_link[i].rotation() << std::endl;
        // print it as quaternion
        Eigen::Quaterniond q(end_effector_link_poses_in_base_link[i].rotation());
        std::cout << "quaternion: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;

        std::cout << "Space Jacobian: " << std::endl;

        Eigen::MatrixXd space_jacobian_of_check_link = space_jacobian_in_base_link[i].transpose();

        // print space jacobian with only active joints
        for (long int j = 0; j < space_jacobian_of_check_link.rows(); j++)
        {
            for (long int k = 0; k < space_jacobian_of_check_link.cols(); k++)
            {
                if (robot_info.getActiveJointMap()[k])
                {
                    std::cout << space_jacobian_of_check_link(j, k) << " ";
                }
            }
            std::cout << std::endl;
        }

        std::cout << "===============================================" << std::endl;

        // use moveit to compute the forward kinematics
        std::vector<double> joint_values_double;
        for (size_t j = 0; j < joint_values_set[i].size(); j++)
        {
            joint_values_double.push_back((double)joint_values_set[i][j]);
        }
        robot_state->setJointGroupPositions(joint_model_group, joint_values_double);
        robot_state->update();
        // Eigen::Isometry3d end_effector_link_pose = robot_state->getGlobalLinkTransform(robot_info.getEndEffectorLinkName());
        Eigen::Isometry3d end_effector_link_pose = robot_state->getGlobalLinkTransform(check_link_name);
        std::cout << "End effector pose " << i << " using moveit: " << std::endl;
        std::cout << "position: " << end_effector_link_pose.translation().transpose() << std::endl;
        // std::cout << end_effector_link_pose.rotation() << std::endl;
        // print it as quaternion
        Eigen::Quaterniond q_moveit(end_effector_link_pose.rotation());
        std::cout << "quaternion: " << q_moveit.w() << " " << q_moveit.x() << " " << q_moveit.y() << " " << q_moveit.z() << std::endl;

        // compute Jacobian
        Eigen::MatrixXd jacobian;
        // robot_state->getJacobian(joint_model_group, robot_state->getLinkModel(robot_info.getEndEffectorLinkName()), Eigen::Vector3d(0, 0, 0), jacobian);
        robot_state->getJacobian(joint_model_group, robot_state->getLinkModel(check_link_name), Eigen::Vector3d(0, 0, 0), jacobian);
        std::cout << "Jacobian: " << std::endl;
        std::cout << jacobian << std::endl;
    }
}

/**
    Use moveit to compute the forward kinematics
 */
void TEST_FORWARD(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    // create moveit robot state
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames()
    );

    std::string check_link_name = robot_info.getEndEffectorLinkName();

    std::vector<std::vector<float>> moveit_positions;
    std::vector<std::vector<float>> moveit_orientations;

    int test_config_num = 100;
    double moveit_update_time = 0.0;
    std::vector<std::vector<float>> joint_values_set;
    for (int i = 0; i < test_config_num; i++)
    {
        // use moveit to randomly sample joint values
        std::vector<double> joint_values_double;
        robot_state->setToRandomPositions(joint_model_group);
        robot_state->copyJointGroupPositions(joint_model_group, joint_values_double);
        std::vector<float> joint_values_float;
        // std::cout << "index " << i << ": ";
        for (size_t j = 0; j < joint_values_double.size(); j++)
        {
            joint_values_float.push_back((float)joint_values_double[j]);
            // std::cout << joint_values_float[j] << " ";
        }
        // std::cout << std::endl;
        joint_values_set.push_back(joint_values_float);

        // store the end effector pose
        auto start = std::chrono::high_resolution_clock::now();
        robot_state->update();
        auto end = std::chrono::high_resolution_clock::now();
        moveit_update_time += std::chrono::duration<double, std::milli>(end - start).count();

        Eigen::Isometry3d end_effector_link_pose = robot_state->getGlobalLinkTransform(check_link_name);
        Eigen::Quaterniond q(end_effector_link_pose.rotation());
        moveit_positions.push_back({
            (float)(end_effector_link_pose.translation().x()),
            (float)(end_effector_link_pose.translation().y()),
            (float)(end_effector_link_pose.translation().z())
        });
        moveit_orientations.push_back({(float)(q.w()), (float)(q.x()), (float)(q.y()), (float)(q.z())});
        // std::cout << "End effector pose " << i << " using moveit: " << std::endl;
        // std::cout << "position: " << end_effector_link_pose.translation().transpose() << std::endl;
        // std::cout << "quaternion: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
    }

    // create states based on the joint values
    auto states = single_arm_space->createStatesFromVector(joint_values_set);
    states->update();

    // statistic_cast_pointer_cast to SingleArmStates
    CUDAMPLib::SingleArmStatesPtr single_arm_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(states);

    std::vector<Eigen::Isometry3d> end_effector_link_poses_in_base_link = single_arm_states->getLinkPoseInBaseLinkHost(check_link_name);

    for (size_t i = 0; i < end_effector_link_poses_in_base_link.size(); i++)
    {
        // std::cout << "End effector pose " << i << ": " << std::endl;
        // std::cout << "position: " << end_effector_link_poses_in_base_link[i].translation().transpose() << std::endl;
        // // std::cout << end_effector_link_poses_in_base_link[i].rotation() << std::endl;
        // // print it as quaternion
        // Eigen::Quaterniond q(end_effector_link_poses_in_base_link[i].rotation());
        // std::cout << "quaternion: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
        Eigen::Quaterniond q(end_effector_link_poses_in_base_link[i].rotation());

        if (fabs(end_effector_link_poses_in_base_link[i].translation().x() - moveit_positions[i][0]) > 0.001 ||
            fabs(end_effector_link_poses_in_base_link[i].translation().y() - moveit_positions[i][1]) > 0.001 ||
            fabs(end_effector_link_poses_in_base_link[i].translation().z() - moveit_positions[i][2]) > 0.001)
        {
            // print in red
            std::cout << "\033[1;31m" << "Error in position at index " << i << "\033[0m" << std::endl;
            continue;
        }

        if (fabs((float)(q.w()) - moveit_orientations[i][0]) > 0.01 ||
            fabs((float)(q.x()) - moveit_orientations[i][1]) > 0.01 ||
            fabs((float)(q.y()) - moveit_orientations[i][2]) > 0.01 ||
            fabs((float)(q.z()) - moveit_orientations[i][3]) > 0.01)
        {
            // print in red
            std::cout << "\033[1;31m" << "Error in orientation at index " << i << "\033[0m" << std::endl;
            continue;
        }

        // print in green
        std::cout << "\033[1;32m" << "Same poses at index " << i << "\033[0m" << std::endl;
    }

    // print the average time for moveit update
    std::cout << "\033[1;32m" << "Average time for moveit update: " << moveit_update_time / (double)test_config_num << " ms" << "\033[0m" << std::endl;

}

void EVAL_FORWARD(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    // create moveit robot state
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames()
    );

    // sample a set of states
    int num_of_test_states = 10000;
    CUDAMPLib::SingleArmStatesPtr single_arm_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(num_of_test_states));

    // update states
    auto start_time = std::chrono::high_resolution_clock::now();
    // single_arm_states->calculateForwardKinematics();
    single_arm_states->calculateForwardKinematicsNvrtv();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "\033[1;32m" << "Time taken by forward kinematics: " << elapsed_time.count() << " seconds" << "\033[0m" << std::endl;
}

void TEST_COLLISION(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    
    // // create obstacles manually
    // balls_pos.push_back({0.4, 0.0, 1.4});
    // ball_radius.push_back(0.2);

    // create obstacles randomly
    generate_sphere_obstacles(balls_pos, ball_radius, group_name, 20, 0.06);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::EnvConstraintSpherePtr env_constraint_sphere = std::make_shared<CUDAMPLib::EnvConstraintSphere>(
        "sphere_obstacle_constraint",
        balls_pos,
        ball_radius
    );
    constraints.push_back(env_constraint_sphere);

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames(),
        0.02f
    );

    int num_of_test_states = 1000;

    // sample a set of states
    CUDAMPLib::SingleArmStatesPtr single_arm_states_1 = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(num_of_test_states));
    if (single_arm_states_1 == nullptr)
    {
        RCLCPP_ERROR(LOGGER, "Failed to sample states for single arm space 1");
        return;
    }
    CUDAMPLib::SingleArmStatesPtr single_arm_states_2 = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(num_of_test_states));
    if (single_arm_states_2 == nullptr)
    {
        RCLCPP_ERROR(LOGGER, "Failed to sample states for single arm space 2");
        return;
    }
    CUDAMPLib::SingleArmStatesPtr single_arm_states_3 = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(num_of_test_states));
    if (single_arm_states_3 == nullptr)
    {
        RCLCPP_ERROR(LOGGER, "Failed to sample states for single arm space 3");
        return;
    }


    auto start_time_update = std::chrono::high_resolution_clock::now();
    single_arm_states_2->update();
    auto end_time_update = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time_update = end_time_update - start_time_update;
    // print in green color
    printf("\033[1;32m" "Time taken by update: %f seconds" "\033[0m \n", elapsed_time_update.count());

    auto start_time_check_states = std::chrono::high_resolution_clock::now();
    single_arm_space->checkStates(single_arm_states_2);
    auto end_time_check_states = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time_check_states = end_time_check_states - start_time_check_states;
    // print in green color
    printf("\033[1;32m" "Time taken by checkStates: %f seconds" "\033[0m \n", elapsed_time_check_states.count());

    // // check motions
    // std::vector<bool> motion_feasibility;
    // std::vector<float> motion_costs;

    // auto start_time_check_motions = std::chrono::high_resolution_clock::now();
    // single_arm_space->checkMotions(single_arm_states_1, single_arm_states_2, motion_feasibility, motion_costs);
    // auto end_time_check_motions = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed_time_check_motions = end_time_check_motions - start_time_check_motions;
    // // print in green color
    // printf("\033[1;32m" "Time taken by checkMotions: %f seconds" "\033[0m \n", elapsed_time_check_motions.count());
}

void TEST_CONSTRAINT_PROJECT(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;

    // create obstacles randomly
    generate_sphere_obstacles(balls_pos, ball_radius, group_name, 20, 0.06);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    // CUDAMPLib::EnvConstraintSpherePtr env_constraint = std::make_shared<CUDAMPLib::EnvConstraintSphere>(
    //     "sphere_obstacle_constraint",
    //     balls_pos,
    //     ball_radius
    // );
    // constraints.push_back(env_constraint);

    int task_link_index = -1;
    for (size_t i = 0; i < robot_info.getLinkNames().size(); i++)
    {
        if (robot_info.getLinkNames()[i] == robot_info.getEndEffectorLinkName())
        {
            task_link_index = i;
            break;
        }
    }

    if (task_link_index == -1)
    {
        RCLCPP_ERROR(LOGGER, "Failed to find the task link index");
        return;
    }

    // create task space constraint
    std::vector<float> reference_frame = {0.9, 0.0, 0.7, 0.0, 0.0, 0.0};
    std::vector<float> tolerance = {0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001};
    CUDAMPLib::TaskSpaceConstraintPtr task_space_constraint = std::make_shared<CUDAMPLib::TaskSpaceConstraint>(
        "task_space_constraint",
        task_link_index,
        Eigen::Matrix4d::Identity(),
        reference_frame,
        tolerance
    );
    constraints.push_back(task_space_constraint);

    // create boundary constraint
    CUDAMPLib::BoundaryConstraintPtr boundary_constraint = std::make_shared<CUDAMPLib::BoundaryConstraint>(
        "boundary_constraint",
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getActiveJointMap()
    );
    constraints.push_back(boundary_constraint);

    // create self collision constraint
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames(),
        0.02f
    );

    int num_of_test_states = 1000;

    // sample a set of states and run ik solver with collision free check
    auto start_time = std::chrono::high_resolution_clock::now();
    CUDAMPLib::SingleArmStatesPtr single_arm_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(num_of_test_states));
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "\033[1;32m" << "Time taken by sample and ik solve: " << elapsed_time.count() << " seconds" << "\033[0m" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    single_arm_states->update();
    std::vector<bool> state_feasibility;
    single_arm_space->checkStates(single_arm_states, state_feasibility);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = end_time - start_time;
    std::cout << "\033[1;32m" << "Time taken by update and checkStates: " << elapsed_time.count() << " seconds" << "\033[0m" << std::endl;

    // visualize the states
    std::vector<std::string> display_links_names = robot_info.getLinkNames();

    std::vector<std::vector<float>> states_joint_values = single_arm_states->getJointStatesHost();

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    // create color
    std_msgs::msg::ColorRGBA color_sample;
    color_sample.r = 0.0;
    color_sample.g = 1.0;
    color_sample.b = 0.0;
    color_sample.a = 0.4;
    visualization_msgs::msg::MarkerArray sample_group_state_markers;
    generate_state_markers(states_joint_values, joint_model_group, robot_state, "sample_group", color_sample, sample_group_state_markers, robot_info.getEndEffectorLinkNames());
    
    // create marker publisher
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("/ik_solver_markers", 10);

    // publish the markers
    while (rclcpp::ok())
    {
        marker_publisher->publish(sample_group_state_markers);
        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    robot_state.reset();
}

void TEST_TASK_WITH_GOAL_REGION(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    int task_link_index = -1;
    for (size_t i = 0; i < robot_info.getLinkNames().size(); i++)
    {
        if (robot_info.getLinkNames()[i] == robot_info.getEndEffectorLinkName())
        {
            task_link_index = i;
            break;
        }
    }

    if (task_link_index == -1)
    {
        RCLCPP_ERROR(LOGGER, "Failed to find the task link index");
        return;
    }

    // create task space constraint
    std::vector<float> reference_frame = {0.9, 0.0, 0.7, 0.0, 0.0, 0.0};
    std::vector<float> tolerance = {1000, 1000, 1000, 0.0001, 0.0001, 0.0001};
    CUDAMPLib::TaskSpaceConstraintPtr task_space_constraint = std::make_shared<CUDAMPLib::TaskSpaceConstraint>(
        "task_space_constraint",
        task_link_index,
        Eigen::Matrix4d::Identity(),
        reference_frame,
        tolerance
    );
    constraints.push_back(task_space_constraint);

    // create boundary constraint
    CUDAMPLib::BoundaryConstraintPtr boundary_constraint = std::make_shared<CUDAMPLib::BoundaryConstraint>(
        "boundary_constraint",
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getActiveJointMap()
    );
    constraints.push_back(boundary_constraint);

    // create self collision constraint
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::BaseSpacePtr goal_region = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames(),
        0.02f
    );

    // start state
    std::vector<float> start_joint_values = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // start state set
    std::vector<std::vector<float>> start_joint_values_set;
    start_joint_values_set.push_back(start_joint_values);

    // create task with goal region
    CUDAMPLib::SingleArmTaskPtr task = std::make_shared<CUDAMPLib::SingleArmTask>(
        start_joint_values_set,
        goal_region
    );

    // visualize the states
    std::vector<std::string> display_links_names = robot_info.getLinkNames();
    std::vector<std::vector<float>> states_joint_values = task->getGoalStatesVector();

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    visualization_msgs::msg::MarkerArray sample_group_state_markers;
    // create color
    std_msgs::msg::ColorRGBA color_sample;
    color_sample.r = 0.0;
    color_sample.g = 1.0;
    color_sample.b = 0.0;
    color_sample.a = 0.4;
    generate_state_markers(states_joint_values, joint_model_group, robot_state, "sample_group", color_sample, sample_group_state_markers, robot_info.getEndEffectorLinkNames());

    // create marker publisher
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("/goal_region_markers", 10);

    // publish the markers
    while (rclcpp::ok())
    {
        marker_publisher->publish(sample_group_state_markers);
        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    robot_state.reset();    
}

/**
    Test filter states
 */
void TEST_FILTER_STATES(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames()
    );

    // sample a set of states
    int num_of_test_states = 100000;
    CUDAMPLib::SingleArmStatesPtr single_arm_states_1 = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(num_of_test_states));
    single_arm_states_1->update();

    // generate filter mask to filter state anlteratively
    std::vector<bool> filter_mask(num_of_test_states, true);

    for (size_t i = 0; i < filter_mask.size(); i++)
    {
        if (i % 2 == 0)
        {
            filter_mask[i] = false;
        }
    }

    // filter states
    auto start_time_filter = std::chrono::high_resolution_clock::now();
    single_arm_states_1->filterStates(filter_mask);
    auto end_time_filter = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time_filter = end_time_filter - start_time_filter;
    // print in green
    std::cout << "\033[1;32m" << "Time taken by filter state function: " << elapsed_time_filter.count() << " seconds" << "\033[0m" << std::endl;
}


void TEST_NEAREST_NEIGHBOR(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames()
    );

    // sample a set of states
    int num_of_test_states = 100;
    CUDAMPLib::SingleArmStatesPtr single_arm_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(num_of_test_states));

    int num_of_query_states = 10;
    CUDAMPLib::SingleArmStatesPtr query_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(num_of_query_states));

    std::vector<int> search_group_1;
    std::vector<int> search_group_2;
    for (int i = 0; i < num_of_test_states; i++)
    {
        if (i % 2 == 0)
            search_group_2.push_back(i);
        else
            search_group_1.push_back(i);
    }
    
    std::vector<std::vector<float>> query_state_joint_values = query_states->getJointStatesHost();
    std::vector<std::vector<float>> manager_state_joint_value = single_arm_states->getJointStatesHost();

    for (size_t i = 0; i < query_state_joint_values.size(); i ++ )
    {
        int nearest_neighbor_index = -1;
        float nearest_neighbor_dis = std::numeric_limits<float>::max();

        for (size_t j = 0; j < search_group_1.size(); j++){
            float sq_dis = 0.0;
            std::vector<float> selected_joint_values = manager_state_joint_value[search_group_1[j]];

            for (size_t k = 0; k < selected_joint_values.size(); k++ )
            {
                sq_dis += (query_state_joint_values[i][k] - selected_joint_values[k]) * (query_state_joint_values[i][k] - selected_joint_values[k]);
            }

            float dis = sqrt(sq_dis);

            if (nearest_neighbor_dis > dis){
                nearest_neighbor_index = j;
                nearest_neighbor_dis = dis;
            }

        }
        std::cout << "query state " << i << " with its nearest neighbor_index " << nearest_neighbor_index << " in group 0 " << std::endl;
    }

    for (size_t i = 0; i < query_state_joint_values.size(); i ++ )
    {
        int nearest_neighbor_index = -1;
        float nearest_neighbor_dis = std::numeric_limits<float>::max();

        for (size_t j = 0; j < search_group_2.size(); j++){
            float sq_dis = 0.0;
            std::vector<float> selected_joint_values = manager_state_joint_value[search_group_2[j]];

            for (size_t k = 0; k < selected_joint_values.size(); k++ )
            {
                sq_dis += (query_state_joint_values[i][k] - selected_joint_values[k]) * (query_state_joint_values[i][k] - selected_joint_values[k]);
            }

            float dis = sqrt(sq_dis);

            if (nearest_neighbor_dis > dis){
                nearest_neighbor_index = j;
                nearest_neighbor_dis = dis;
            }

        }
        std::cout << "query state " << i << " with its nearest neighbor_index " << nearest_neighbor_index << " in group 1 " << std::endl;
    }

    // create state manager
    auto state_manager = single_arm_space->createStateManager();

    // add sampled states to state manager
    state_manager->add_states(single_arm_states);

    std::vector<std::vector<int>> index_groups;
    index_groups.push_back(search_group_1);
    index_groups.push_back(search_group_2);

    std::vector<std::vector<int>> neighbors_index;

    state_manager->find_the_nearest_neighbors(
        query_states, 
        index_groups,
        neighbors_index
    );

    std::cout << "---------------------------------------------" << std::endl;
    for(size_t j = 0; j < index_groups.size(); j++)
    {
        for(int i = 0; i < query_states->getNumOfStates(); i++)
        {
            std::cout << "query state " << i << " with its nearest neighbor_index " << neighbors_index[i][j] << " in group " << j << std::endl;
        }
    }
}


/**
    Create a CUDAMPLib::SingleArmSpace and sample a set of states.
    Then, we will check the feasibility of the states and visualize the collision spheres in rviz.
 */
void TEST_COLLISION_AND_VIS(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    /***************************** Generate Obstacles **************************************************/

    // find the region where the obstacles should not be placed
    std::vector<BoundingBox> unmoveable_bounding_boxes_of_robot = getUnmoveableBoundingBoxes(robot_model, group_name, 0.05);

    // create obstacles for spheres
    std::vector<Sphere> collision_spheres;
    genSphereObstacles(20, 0.08, 0.06, unmoveable_bounding_boxes_of_robot, collision_spheres);

    // create obstacles for cuboids
    std::vector<BoundingBox> bounding_boxes;
    genCuboidObstacles(20, 0.3, 0.05, unmoveable_bounding_boxes_of_robot, bounding_boxes);

    // create obstacles for cylinders
    std::vector<Cylinder> cylinders;
    genCylinderObstacles(20, 0.08, 0.05, 0.8, 0.1, unmoveable_bounding_boxes_of_robot, cylinders);


    // convert to vector of vector so we can pass it to CUDAMPLib::EnvConstraintSphere
    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    SphereToVectors(collision_spheres, balls_pos, ball_radius);

    // convert to vector of vector so we can pass it to CUDAMPLib::EnvConstraintCuboid
    std::vector<std::vector<float>> bounding_boxes_pos;
    std::vector<std::vector<float>> bounding_boxes_orientation_matrix;
    std::vector<std::vector<float>> bounding_boxes_max;
    std::vector<std::vector<float>> bounding_boxes_min;
    CuboidToVectors(bounding_boxes, bounding_boxes_pos, bounding_boxes_orientation_matrix, bounding_boxes_max, bounding_boxes_min);

    // convert to vector of vector so we can pass it to CUDAMPLib::EnvConstraintCylinder
    std::vector<std::vector<float>> cylinders_pos;
    std::vector<std::vector<float>> cylinders_orientation_matrix;
    std::vector<float> cylinders_radius;
    std::vector<float> cylinders_height;
    CylinderToVectors(cylinders, cylinders_pos, cylinders_orientation_matrix, cylinders_radius, cylinders_height);

    // generate the markers for the obstacles
    visualization_msgs::msg::MarkerArray obstacle_collision_spheres_marker_array = generateSpheresMarkers(collision_spheres, node);
    visualization_msgs::msg::MarkerArray obstacle_collision_cuboids_marker_array = generateBoundingBoxesMarkers(bounding_boxes, node);
    visualization_msgs::msg::MarkerArray obstacle_collision_cylinders_marker_array = generateCylindersMarkers(cylinders, node);


    /********************** Generate env constraint ********************************/

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    // Create obstacle constraint for sphere
    CUDAMPLib::EnvConstraintSpherePtr env_constraint_sphere = std::make_shared<CUDAMPLib::EnvConstraintSphere>(
        "sphere_obstacle_constraint",
        balls_pos,
        ball_radius
    );
    constraints.push_back(env_constraint_sphere);

    // Create obstacle constraint for cuboid
    CUDAMPLib::EnvConstraintCuboidPtr env_constraint_cuboid = std::make_shared<CUDAMPLib::EnvConstraintCuboid>(
        "cuboid_bstacle_constraint",
        bounding_boxes_pos,
        bounding_boxes_orientation_matrix,
        bounding_boxes_max,
        bounding_boxes_min
    );
    constraints.push_back(env_constraint_cuboid);

    // Create obstacle constraint for cylinder
    CUDAMPLib::EnvConstraintCylinderPtr env_constraint_cylinder = std::make_shared<CUDAMPLib::EnvConstraintCylinder>(
        "cylinder_obstacle_constraint",
        cylinders_pos,
        cylinders_orientation_matrix,
        cylinders_radius,
        cylinders_height
    );
    constraints.push_back(env_constraint_cylinder);

    /*******************************************************************************/

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames()
    );

    // sample a set of states
    CUDAMPLib::SingleArmStatesPtr sampled_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(20));
    sampled_states->update();

    std::vector<bool> state_feasibility;

    // check states
    single_arm_space->checkStates(sampled_states, state_feasibility);

    std::vector<std::string> display_links_names = joint_model_group->getLinkModelNames();
    
    // add gripper link
    for(std::string end_effector_link : robot_info.getEndEffectorLinkNames())
    {
        display_links_names.push_back(end_effector_link);
    }

    std::vector<visualization_msgs::msg::MarkerArray> success_sample_group_state_markers;
    std::vector<visualization_msgs::msg::MarkerArray> fail_sample_group_state_markers;

    std::vector<std::vector<float>> states_joint_values = sampled_states->getJointStatesFullHost();
    for (size_t i = 0; i < states_joint_values.size(); i++)
    {
        std::vector<double> states_joint_values_i_double;
        for (size_t j = 0; j < states_joint_values[i].size(); j++)
        {
            // print only active joints
            if (robot_info.getActiveJointMap()[j])
            {
                states_joint_values_i_double.push_back((double)states_joint_values[i][j]);
            }
        }

        robot_state->setJointGroupPositions(joint_model_group, states_joint_values_i_double);
        robot_state->update();

        visualization_msgs::msg::MarkerArray robot_marker;
        // color
        std_msgs::msg::ColorRGBA color;
        if (state_feasibility[i])
        {
            color.r = 0.0;
            color.g = 1.0;
            color.b = 0.0;
            color.a = 0.4;
            const std::string sample_group_ns = "success_sampled_group";
            robot_state->getRobotMarkers(robot_marker, display_links_names, color, sample_group_ns, rclcpp::Duration::from_seconds(0));
            success_sample_group_state_markers.push_back(robot_marker);
        }
        else
        {
            color.r = 1.0;
            color.g = 0.0;
            color.b = 0.0;
            color.a = 0.4;
            const std::string sample_group_ns = "fail_sampled_group";
            robot_state->getRobotMarkers(robot_marker, display_links_names, color, sample_group_ns, rclcpp::Duration::from_seconds(0));
            fail_sample_group_state_markers.push_back(robot_marker);
        }
        
    }

    visualization_msgs::msg::MarkerArray success_sample_group_state_markers_combined;
    for (size_t i = 0; i < success_sample_group_state_markers.size(); i++)
    {
        success_sample_group_state_markers_combined.markers.insert(
            success_sample_group_state_markers_combined.markers.end(), 
            success_sample_group_state_markers[i].markers.begin(), success_sample_group_state_markers[i].markers.end());
    }

    // update the id
    for (size_t i = 0; i < success_sample_group_state_markers_combined.markers.size(); i++)
    {
        success_sample_group_state_markers_combined.markers[i].id = i;
    }

    visualization_msgs::msg::MarkerArray fail_sample_group_state_markers_combined;
    for (size_t i = 0; i < fail_sample_group_state_markers.size(); i++)
    {
        fail_sample_group_state_markers_combined.markers.insert(
            fail_sample_group_state_markers_combined.markers.end(), 
            fail_sample_group_state_markers[i].markers.begin(), fail_sample_group_state_markers[i].markers.end());
    }

    // update the id
    for (size_t i = 0; i < fail_sample_group_state_markers_combined.markers.size(); i++)
    {
        fail_sample_group_state_markers_combined.markers[i].id = i;
    }

    std::vector<std::vector<std::vector<float>>> self_collision_spheres_pos =  sampled_states->getSelfCollisionSpheresPosInBaseLinkHost();

    std::vector<std::vector<float>> collision_spheres_pos_of_selected_config = self_collision_spheres_pos[0];

    /*======================================= prepare publishers ==================================================================== */

    // Create marker publisher
    auto self_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("self_collision_spheres", 1);
    auto sphere_obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);
    auto cuboid_obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_cuboids", 1);
    auto cylinder_obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_cylinders", 1);
    auto success_sample_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("success_sample_group_states", 1);
    auto fail_sample_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("fail_sample_group_states", 1);
    // Create a self MarkerArray message
    visualization_msgs::msg::MarkerArray robot_collision_spheres_marker_array = generate_self_collision_markers(
        collision_spheres_pos_of_selected_config,
        robot_info.getCollisionSpheresRadius(),
        node
    );

    // use loop to publish the trajectory
    while (rclcpp::ok())
    {
        // Publish the message
        self_marker_publisher->publish(robot_collision_spheres_marker_array);
        success_sample_group_states_publisher->publish(success_sample_group_state_markers_combined);
        fail_sample_group_states_publisher->publish(fail_sample_group_state_markers_combined);
        sphere_obstacle_marker_publisher->publish(obstacle_collision_spheres_marker_array);
        cuboid_obstacle_marker_publisher->publish(obstacle_collision_cuboids_marker_array);
        cylinder_obstacle_marker_publisher->publish(obstacle_collision_cylinders_marker_array);
        
        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    robot_state.reset();
}

/**
    We will generate a random start and goal states, then use the CUDAMPLib::RRG to plan a path between them.
    The planned path will be visualized in rviz.
 */
void generateRandomStartAndGoal(
    moveit::core::RobotStatePtr & robot_state,
    const moveit::core::JointModelGroup* joint_model_group,
    const planning_scene::PlanningScenePtr & planning_scene,
    const std::string & group_name,
    std::vector<float> & start_joint_values,
    std::vector<float> & goal_joint_values,
    moveit_msgs::msg::RobotState & start_state_msg,
    moveit_msgs::msg::RobotState & goal_state_msg
)
{
    // generate random state on the joint model group
    robot_state->setToRandomPositions(joint_model_group);
    robot_state->update();

    while(not planning_scene->isStateValid(*robot_state, group_name))
    {
        robot_state->setToRandomPositions(joint_model_group);
        robot_state->update();
    }

    std::vector<double> start_joint_values_double;
    robot_state->copyJointGroupPositions(joint_model_group, start_joint_values_double);
    for (size_t i = 0; i < start_joint_values_double.size(); i++)
    {
        start_joint_values.push_back((float)start_joint_values_double[i]);
    }

    // set start state
    
    moveit::core::robotStateToRobotStateMsg(*robot_state, start_state_msg);
    
    // generate random state on the joint model group
    robot_state->setToRandomPositions(joint_model_group);
    robot_state->update();

    while(not planning_scene->isStateValid(*robot_state, group_name))
    {
        robot_state->setToRandomPositions(joint_model_group);
        robot_state->update();
    }

    std::vector<double> goal_joint_values_double;
    robot_state->copyJointGroupPositions(joint_model_group, goal_joint_values_double);
    for (size_t i = 0; i < goal_joint_values_double.size(); i++)
    {
        goal_joint_values.push_back((float)goal_joint_values_double[i]);
    }

    // set goal state
    moveit::core::robotStateToRobotStateMsg(*robot_state, goal_state_msg);

    // print "start and goal state"
    std::cout << "start state: ";
    for (size_t i = 0; i < start_joint_values.size(); i++)
    {
        std::cout << start_joint_values[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "goal state: ";
    for (size_t i = 0; i < goal_joint_values.size(); i++)
    {
        std::cout << goal_joint_values[i] << " ";
    }
    std::cout << std::endl;
}

void TEST_Planner(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    /***************************** 1. Prepare Robot information **************************************************/
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);
    // set robot state to default state
    robot_state->setToDefaultValues();
    robot_state->update();

    /***************************** 2. Generate Obstacles **************************************************/

    // find the region where the obstacles should not be placed
    std::vector<BoundingBox> unmoveable_bounding_boxes_of_robot = getUnmoveableBoundingBoxes(robot_model, group_name, 0.05);

    // create obstacles for spheres
    std::vector<Sphere> collision_spheres;
    genSphereObstacles(1, 0.08, 0.06, unmoveable_bounding_boxes_of_robot, collision_spheres);

    // create obstacles for cuboids
    std::vector<BoundingBox> bounding_boxes;
    genCuboidObstacles(1, 0.05, 0.05, unmoveable_bounding_boxes_of_robot, bounding_boxes);

    // create obstacles for cylinders
    std::vector<Cylinder> cylinders;
    genCylinderObstacles(1, 0.08, 0.05, 0.8, 0.1, unmoveable_bounding_boxes_of_robot, cylinders);


    // convert to vector of vector so we can pass it to CUDAMPLib::EnvConstraintSphere
    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    SphereToVectors(collision_spheres, balls_pos, ball_radius);

    // convert to vector of vector so we can pass it to CUDAMPLib::EnvConstraintCuboid
    std::vector<std::vector<float>> bounding_boxes_pos;
    std::vector<std::vector<float>> bounding_boxes_orientation_matrix;
    std::vector<std::vector<float>> bounding_boxes_max;
    std::vector<std::vector<float>> bounding_boxes_min;
    CuboidToVectors(bounding_boxes, bounding_boxes_pos, bounding_boxes_orientation_matrix, bounding_boxes_max, bounding_boxes_min);

    // convert to vector of vector so we can pass it to CUDAMPLib::EnvConstraintCylinder
    std::vector<std::vector<float>> cylinders_pos;
    std::vector<std::vector<float>> cylinders_orientation_matrix;
    std::vector<float> cylinders_radius;
    std::vector<float> cylinders_height;
    CylinderToVectors(cylinders, cylinders_pos, cylinders_orientation_matrix, cylinders_radius, cylinders_height);

    // generate the markers for the obstacles
    visualization_msgs::msg::MarkerArray obstacle_collision_spheres_marker_array = generateSpheresMarkers(collision_spheres, node);
    visualization_msgs::msg::MarkerArray obstacle_collision_cuboids_marker_array = generateBoundingBoxesMarkers(bounding_boxes, node);
    visualization_msgs::msg::MarkerArray obstacle_collision_cylinders_marker_array = generateCylindersMarkers(cylinders, node);

    /***************************** 3. Generate Start and Goal States **************************************************/

    // create planning scene
    auto world = std::make_shared<collision_detection::World>();
    auto planning_scene = std::make_shared<planning_scene::PlanningScene>(robot_model, world);

    // Add spheres as obstacles to the planning scene
    for (size_t i = 0; i < collision_spheres.size(); i++)
    {
        Eigen::Isometry3d sphere_pose = Eigen::Isometry3d::Identity();
        sphere_pose.translation() = Eigen::Vector3d(collision_spheres[i].x, collision_spheres[i].y, collision_spheres[i].z);
        planning_scene->getWorldNonConst()->addToObject("obstacle_" + std::to_string(i), shapes::ShapeConstPtr(new shapes::Sphere(collision_spheres[i].radius)), sphere_pose);
    }

    // Add cuboids as obstacles to the planning scene
    for (size_t i = 0; i < bounding_boxes.size(); i++)
    {
        // Get the current bounding box
        BoundingBox box = bounding_boxes[i];

        // Compute the dimensions of the cuboid
        float dim_x = box.x_max - box.x_min;
        float dim_y = box.y_max - box.y_min;
        float dim_z = box.z_max - box.z_min;

        // Create the Box shape using the dimensions
        shapes::ShapeConstPtr box_shape(new shapes::Box(dim_x, dim_y, dim_z));

        // Compute the center of the box in its local coordinate frame.
        // Note: this center is relative to the reference point given by box.x, box.y, box.z.
        float center_local_x = (box.x_min + box.x_max) / 2.0;
        float center_local_y = (box.y_min + box.y_max) / 2.0;
        float center_local_z = (box.z_min + box.z_max) / 2.0;
        Eigen::Vector3d center_local(center_local_x, center_local_y, center_local_z);

        // Build the rotation from roll, pitch, yaw.
        Eigen::Quaterniond quat;
        quat = Eigen::AngleAxisd(box.roll, Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(box.pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(box.yaw, Eigen::Vector3d::UnitZ());

        // The provided pose (box.x, box.y, box.z) is not at the center,
        // so compute the final translation by adding the rotated local center offset.
        Eigen::Vector3d pose_translation(box.x, box.y, box.z);
        Eigen::Isometry3d box_pose = Eigen::Isometry3d::Identity();
        box_pose.linear() = quat.toRotationMatrix();
        box_pose.translation() = pose_translation + quat * center_local;

        // Add the box as an obstacle to the planning scene.
        planning_scene->getWorldNonConst()->addToObject("obstacle_box_" + std::to_string(i),
                                                        box_shape, box_pose);
    }

    // Add cylinders as obstacles to the planning scene
    for (size_t i = 0; i < cylinders.size(); i++)
    {
        // Create an identity transformation for the cylinder pose
        Eigen::Isometry3d cylinder_pose = Eigen::Isometry3d::Identity();

        // Set the translation using the cylinder's x, y, z coordinates
        cylinder_pose.translation() = Eigen::Vector3d(cylinders[i].x, 
                                                    cylinders[i].y, 
                                                    cylinders[i].z);

        // Compute the orientation from roll, pitch, and yaw using Eigen's AngleAxis
        Eigen::Quaterniond q = Eigen::AngleAxisd(cylinders[i].roll, Eigen::Vector3d::UnitX()) *
                            Eigen::AngleAxisd(cylinders[i].pitch, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(cylinders[i].yaw, Eigen::Vector3d::UnitZ());
        cylinder_pose.rotate(q);

        // Create the cylinder shape and add it to the planning scene
        planning_scene->getWorldNonConst()->addToObject(
            "obstacle_cylinder_" + std::to_string(i),
            shapes::ShapeConstPtr(new shapes::Cylinder(cylinders[i].height, cylinders[i].radius)),
            cylinder_pose);
    }

    // generate start and goal states
    std::vector<float> start_joint_values;
    std::vector<float> goal_joint_values;
    moveit_msgs::msg::RobotState start_state_msg;
    moveit_msgs::msg::RobotState goal_state_msg;
    generateRandomStartAndGoal(robot_state, joint_model_group, planning_scene, group_name, start_joint_values, goal_joint_values, start_state_msg, goal_state_msg);

    std::cout << ">>>>>>>>>>> Generate random start and goal: Done" << std::endl;

    std::vector<std::vector<float>> start_joint_values_set;
    start_joint_values_set.push_back(start_joint_values);
    std::vector<std::vector<float>> goal_joint_values_set;
    goal_joint_values_set.push_back(goal_joint_values);

    // create the task from start and goal states
    CUDAMPLib::SingleArmTaskPtr task = std::make_shared<CUDAMPLib::SingleArmTask>(
        start_joint_values_set,
        goal_joint_values_set
    );

    // prepare the state markers for both start and goal state to visualize later
    moveit_msgs::msg::DisplayRobotState start_display_robot_state;
    start_display_robot_state.state = start_state_msg;
    moveit_msgs::msg::DisplayRobotState goal_display_robot_state;
    goal_display_robot_state.state = goal_state_msg;

    /***************************** 4. Create Constraints For Planning **************************************************/

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    // Create obstacle constraint for sphere
    CUDAMPLib::EnvConstraintSpherePtr env_constraint_sphere = std::make_shared<CUDAMPLib::EnvConstraintSphere>(
        "sphere_obstacle_constraint",
        balls_pos,
        ball_radius
    );
    constraints.push_back(env_constraint_sphere);

    // Create obstacle constraint for cuboid
    CUDAMPLib::EnvConstraintCuboidPtr env_constraint_cuboid = std::make_shared<CUDAMPLib::EnvConstraintCuboid>(
        "cuboid_obstacle_constraint",
        bounding_boxes_pos,
        bounding_boxes_orientation_matrix,
        bounding_boxes_max,
        bounding_boxes_min
    );
    constraints.push_back(env_constraint_cuboid);

    // Create obstacle constraint for cylinder
    CUDAMPLib::EnvConstraintCylinderPtr env_constraint_cylinder = std::make_shared<CUDAMPLib::EnvConstraintCylinder>(
        "cylinder_obstacle_constraint",
        cylinders_pos,
        cylinders_orientation_matrix,
        cylinders_radius,
        cylinders_height
    );
    constraints.push_back(env_constraint_cylinder);
    
    // Create self collision constraint
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    /****************************** 5. Create Space *******************************************************************/

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames(),
        0.02 // resolution
    );

    /****************************** 6. Create Planner and set planning parameters *******************************************************/

    // // create the planner
    // CUDAMPLib::RRGPtr planner = std::make_shared<CUDAMPLib::RRG>(single_arm_space);

    // // set planner parameters
    // planner->setMaxTravelDistance(5.0);
    // planner->setSampleAttemptsInEachIteration(30);
    
    // // set the task
    // planner->setMotionTask(task);

    // // create termination condition
    // // CUDAMPLib::StepTerminationPtr termination_condition = std::make_shared<CUDAMPLib::StepTermination>(10);
    // CUDAMPLib::TimeoutTerminationPtr termination_condition = std::make_shared<CUDAMPLib::TimeoutTermination>(10.0);

    /*********************************************************************************************************************************** */
    // create the planner
    CUDAMPLib::cRRTCPtr planner = std::make_shared<CUDAMPLib::cRRTC>(single_arm_space);

    // set the task
    planner->setMotionTask(task);

    // create termination condition
    CUDAMPLib::StepTerminationPtr termination_condition = std::make_shared<CUDAMPLib::StepTermination>(1);
    // CUDAMPLib::TimeoutTerminationPtr termination_condition = std::make_shared<CUDAMPLib::TimeoutTermination>(10.0);

    /****************************** 7. Solve the task ********************************************************************************/

    // solve the task
    auto start_time = std::chrono::high_resolution_clock::now();
    planner->solve(termination_condition);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "\033[1;32m" << "Time taken by function: " << elapsed_time.count() << " seconds" << "\033[0m" << std::endl;

    /************************** 8. Visualize both start and goal group **************************************/

    // // extract the start and goal group states
    // CUDAMPLib::BaseStatesPtr start_group_states;
    // CUDAMPLib::BaseStatesPtr goal_group_states;
    // planner->getStartAndGoalGroupStates(start_group_states, goal_group_states);

    // // static_pointer_cast to SingleArmStates
    // CUDAMPLib::SingleArmStatesPtr start_group_states_single_arm = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(start_group_states);
    // CUDAMPLib::SingleArmStatesPtr goal_group_states_single_arm = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(goal_group_states);

    // // create color for start group
    // std_msgs::msg::ColorRGBA color_start;
    // color_start.r = 0.0;
    // color_start.g = 1.0;
    // color_start.b = 0.0;
    // color_start.a = 0.4;

    // // visualize the start group states
    // visualization_msgs::msg::MarkerArray start_group_state_markers_combined;
    // generate_state_markers(
    //     start_group_states_single_arm->getJointStatesHost(),
    //     joint_model_group,
    //     robot_state,
    //     "start_group",
    //     color_start,
    //     start_group_state_markers_combined,
    //     robot_info.getEndEffectorLinkNames()
    // );

    // // create color for goal group
    // std_msgs::msg::ColorRGBA color_goal;
    // color_goal.r = 1.0;
    // color_goal.g = 0.0;
    // color_goal.b = 0.0;
    // color_goal.a = 0.4;

    // visualization_msgs::msg::MarkerArray goal_group_state_markers_combined;
    // generate_state_markers(
    //     goal_group_states_single_arm->getJointStatesHost(),
    //     joint_model_group,
    //     robot_state,
    //     "goal_group",
    //     color_goal,
    //     goal_group_state_markers_combined,
    //     robot_info.getEndEffectorLinkNames()
    // );

    // /************************** 9. create the trajectory marker if exists **************************************/

    // moveit_msgs::msg::DisplayTrajectory display_trajectory;
    // moveit_msgs::msg::RobotTrajectory robot_trajectory_msg;
    // auto solution_robot_trajectory = robot_trajectory::RobotTrajectory(robot_model, joint_model_group);

    // if (task->hasSolution())
    // {
    //     // print "Task solved" in green color
    //     std::cout << "\033[1;32m" << "Task solved" << "\033[0m" << std::endl;

    //     std::vector<std::vector<float>> solution_path = task->getSolution();

    //     // generate robot trajectory msg
    //     for (size_t i = 0; i < solution_path.size(); i++)
    //     {
    //         // convert solution_path[i] to double vector
    //         std::vector<double> solution_path_i_double = std::vector<double>(solution_path[i].begin(), solution_path[i].end());
    //         robot_state->setJointGroupPositions(joint_model_group, solution_path_i_double);
    //         solution_robot_trajectory.addSuffixWayPoint(*robot_state, 1.0);
    //     }
    //     // Create a DisplayTrajectory message
    //     solution_robot_trajectory.getRobotTrajectoryMsg(robot_trajectory_msg);

    //     display_trajectory.trajectory_start = start_state_msg;
    //     display_trajectory.trajectory.push_back(robot_trajectory_msg);
    // }
    // else
    // {
    //     // print "Task not solved" in red color
    //     std::cout << "\033[1;31m" << "Task not solved" << "\033[0m" << std::endl;

    //     // print the failure reason
    //     std::cout << "Failure reason: " << task->getFailureReason() << std::endl;
    // }

    // /************************************* 10. prepare publishers ******************************************* */

    // // Create a start robot state publisher
    // auto start_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("start_robot_state", 1);
    // auto goal_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("goal_robot_state", 1);
    // auto sphere_obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);
    // auto cuboid_obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_cuboids", 1);
    // auto cylinder_obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_cylinders", 1);
    // auto display_publisher = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);
    // auto start_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("start_group_states", 1);
    // auto goal_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("goal_group_states", 1);
    
    // /************************************ 11. loop for visulize ************************************************************/

    // // Publish the message in a loop
    // while (rclcpp::ok())
    // {
    //     // Publish the message
    //     start_robot_state_publisher->publish(start_display_robot_state);
    //     goal_robot_state_publisher->publish(goal_display_robot_state);
    //     sphere_obstacle_marker_publisher->publish(obstacle_collision_spheres_marker_array);
    //     cuboid_obstacle_marker_publisher->publish(obstacle_collision_cuboids_marker_array);
    //     cylinder_obstacle_marker_publisher->publish(obstacle_collision_cylinders_marker_array);
    //     start_group_states_publisher->publish(start_group_state_markers_combined);
    //     goal_group_states_publisher->publish(goal_group_state_markers_combined);

    //     if (task->hasSolution())
    //     {
    //         display_publisher->publish(display_trajectory);
    //     }
        
    //     rclcpp::spin_some(node);

    //     // sleep for 1 second
    //     std::this_thread::sleep_for(std::chrono::seconds(1));
    // }

    // clear the robot state
    robot_state.reset();
}

void TEST_OMPL(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    // Prepare obstacle constraint
    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    generate_sphere_obstacles(balls_pos, ball_radius, group_name, 20, 0.06);

    // create planning scene
    auto world = std::make_shared<collision_detection::World>();
    auto planning_scene = std::make_shared<planning_scene::PlanningScene>(robot_model, world);

    // add those balls to the planning scene
    for (size_t i = 0; i < balls_pos.size(); i++)
    {
        Eigen::Isometry3d sphere_pose = Eigen::Isometry3d::Identity();
        sphere_pose.translation() = Eigen::Vector3d(balls_pos[i][0], balls_pos[i][1], balls_pos[i][2]);
        planning_scene->getWorldNonConst()->addToObject("obstacle_" + std::to_string(i), shapes::ShapeConstPtr(new shapes::Sphere(ball_radius[i])), sphere_pose);
    }

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    // generate start and goal states
    std::vector<float> start_joint_values;
    std::vector<float> goal_joint_values;
    moveit_msgs::msg::RobotState start_state_msg;
    moveit_msgs::msg::RobotState goal_state_msg;
    generateRandomStartAndGoal(robot_state, joint_model_group, planning_scene, group_name, start_joint_values, goal_joint_values, start_state_msg, goal_state_msg);

    // set group dimension
    int dim = robot_model->getJointModelGroup(group_name)->getActiveJointModels().size();
    std::cout << "dim: " << dim << std::endl;

    // create ompl states for start and goal
    ompl::base::ScopedState<> start_state(ompl::base::StateSpacePtr(new ompl::base::RealVectorStateSpace(dim)));
    ompl::base::ScopedState<> goal_state(ompl::base::StateSpacePtr(new ompl::base::RealVectorStateSpace(dim)));

    for (int i = 0; i < dim; i++)
    {
        start_state[i] = start_joint_values[i];
        goal_state[i] = goal_joint_values[i];
    }

    // create ompl space
    ompl::base::StateSpacePtr space(new ompl::base::RealVectorStateSpace(dim));

    std::vector<float> upper_bounds_of_active_joints;
    std::vector<float> lower_bounds_of_active_joints;

    // based on active joint map, set the bounds
    for (size_t i = 0; i < robot_info.getActiveJointMap().size(); i++)
    {
        if (robot_info.getActiveJointMap()[i])
        {
            upper_bounds_of_active_joints.push_back(robot_info.getUpperBounds()[i]);
            lower_bounds_of_active_joints.push_back(robot_info.getLowerBounds()[i]);
        }
    }

    // set bounds
    ompl::base::RealVectorBounds bounds(dim);
    for (int i = 0; i < dim; i++)
    {
        bounds.setLow(i, lower_bounds_of_active_joints[i]);
        bounds.setHigh(i, upper_bounds_of_active_joints[i]);
    }
    space->as<ompl::base::RealVectorStateSpace>()->setBounds(bounds);

    // create space information
    ompl::base::SpaceInformationPtr si(new ompl::base::SpaceInformation(space));

    // set state validity checker
    si->setStateValidityChecker([&](const ompl::base::State * state) {
        // convert ompl state to robot state
        std::vector<double> joint_values_double;
        for (int i = 0; i < dim; i++)
        {
            joint_values_double.push_back(state->as<ompl::base::RealVectorStateSpace::StateType>()->values[i]);
            // check if joint value is in the joint limits
            if (joint_values_double[i] < lower_bounds_of_active_joints[i] || joint_values_double[i] > upper_bounds_of_active_joints[i])
            {
                return false;
            }
        }
        robot_state->setJointGroupPositions(joint_model_group, joint_values_double);
        robot_state->update();
        return planning_scene->isStateValid(*robot_state, group_name);
    });

    // set problem definition
    ompl::base::ProblemDefinitionPtr pdef(new ompl::base::ProblemDefinition(si));
    pdef->setStartAndGoalStates(start_state, goal_state);

    // create planner
    auto planner(std::make_shared<og::RRTConnect>(si));
    planner->setProblemDefinition(pdef);
    planner->setup();

    // solve the problem
    ompl::base::PlannerStatus solved = planner->ob::Planner::solve(1.0);

    // generate robot trajectory msg
    moveit_msgs::msg::DisplayTrajectory display_trajectory;
    moveit_msgs::msg::RobotTrajectory robot_trajectory_msg;
    auto solution_robot_trajectory = robot_trajectory::RobotTrajectory(robot_model, joint_model_group);

    if (solved)
    {
        // print "Task solved" in green color
        std::cout << "\033[1;32m" << "Task solved" << "\033[0m" << std::endl;

        // get the path from the planner
        ob::PathPtr path = pdef->getSolutionPath();
        const auto *path_ = path.get()->as<og::PathGeometric>();
        // convert the path to robot trajectory, and do interpolation between states
        const ob::State * state = path_->getState(0);
        std::vector<double> previous_joint_values_double;
        for (int j = 0; j < dim; j++)
        {
            previous_joint_values_double.push_back(state->as<ompl::base::RealVectorStateSpace::StateType>()->values[j]);
        }

        for (size_t i = 1; i < path_->getStateCount(); i++)
        {
            const ob::State * state = path_->getState(i);
            std::vector<double> joint_values_double;
            for (int j = 0; j < dim; j++)
            {
                joint_values_double.push_back(state->as<ompl::base::RealVectorStateSpace::StateType>()->values[j]);
            }
            // interpolate between previous_joint_values_double and joint_values_double
            std::vector<std::vector<double>> interpolated_joint_values_double = interpolate(previous_joint_values_double, joint_values_double, 10);

            // print the interpolated joint values only
            for (size_t k = 0; k < interpolated_joint_values_double.size(); k++)
            {
                robot_state->setJointGroupPositions(joint_model_group, interpolated_joint_values_double[k]);
                solution_robot_trajectory.addSuffixWayPoint(*robot_state, 10.0);
            }

            previous_joint_values_double = joint_values_double;
        }

        // Create a DisplayTrajectory message
        solution_robot_trajectory.getRobotTrajectoryMsg(robot_trajectory_msg);
        display_trajectory.trajectory_start = start_state_msg;
        display_trajectory.trajectory.push_back(robot_trajectory_msg);
    }
    else
    {
        // print "Task not solved" in red color
        std::cout << "\033[1;31m" << "Task not solved" << "\033[0m" << std::endl;
    }

    // Create a start robot state publisher
    auto start_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("start_robot_state", 1);
    // Create a goal robot state publisher
    auto goal_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("goal_robot_state", 1);
    auto obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);
    std::shared_ptr<rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>> display_publisher =
        node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);
    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState start_display_robot_state;
    start_display_robot_state.state = start_state_msg;
    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState goal_display_robot_state;
    goal_display_robot_state.state = goal_state_msg;
    // Create a obstacle MarkerArray message
    visualization_msgs::msg::MarkerArray obstacle_collision_spheres_marker_array = generate_obstacles_markers(balls_pos, ball_radius, node);

    std::cout << "publishing start and goal robot state" << std::endl;
    // Publish the message in a loop
    while (rclcpp::ok())
    {
        // Publish the message
        start_robot_state_publisher->publish(start_display_robot_state);
        goal_robot_state_publisher->publish(goal_display_robot_state);
        obstacle_marker_publisher->publish(obstacle_collision_spheres_marker_array);

        if (solved)
        {
            display_publisher->publish(display_trajectory);
        }
        
        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    robot_state.reset();
}

/**
    @brief This function is used to test the constrained motion planning where the task space constraint
    is ensuring the end effector is horizontal to the ground. It first randomly generate two states satisfying the task space constraint,
    then use the CUDAMPLib::RRG to plan a path between them, and the path must satisfy the task space constraint as well.
 */
void TEST_CONSTRAINED_MOTION_PLANNING(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);
    
    // Prepare constraints
    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;
    
    // Create self collision constraint
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create task space constraint
    int task_link_index = -1;
    for (size_t i = 0; i < robot_info.getLinkNames().size(); i++)
    {
        if (robot_info.getLinkNames()[i] == robot_info.getEndEffectorLinkName())
        {
            task_link_index = i;
            break;
        }
    }
    if (task_link_index == -1)
    {
        RCLCPP_ERROR(LOGGER, "Failed to find the task link index");
        return;
    }

    std::vector<float> reference_frame = {0.9, 0.0, 0.7, 0.0, 0.0, 0.0};
    std::vector<float> tolerance = {1000, 1000, 0.0001, 0.0001, 0.0001, 10};
    CUDAMPLib::TaskSpaceConstraintPtr task_space_constraint = std::make_shared<CUDAMPLib::TaskSpaceConstraint>(
        "task_space_constraint",
        task_link_index,
        Eigen::Matrix4d::Identity(),
        reference_frame,
        tolerance
    );
    constraints.push_back(task_space_constraint);

    // Create boundary constraint
    CUDAMPLib::BoundaryConstraintPtr boundary_constraint = std::make_shared<CUDAMPLib::BoundaryConstraint>(
        "boundary_constraint",
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getActiveJointMap()
    );
    constraints.push_back(boundary_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames(),
        0.02 // resolution
    );

    // generate start and goal states under the task space constraint by sampling
    CUDAMPLib::SingleArmStatesPtr sample_single_arm_states = 
        std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(100));

    // check states
    sample_single_arm_states->update();
    std::vector<bool> state_feasibility;
    single_arm_space->checkStates(sample_single_arm_states, state_feasibility);

    // filter out the infeasible states
    sample_single_arm_states->filterStates(state_feasibility);

    // print states
    std::vector<std::vector<float>> joint_values = sample_single_arm_states->getJointStatesHost();

    // if the number of states is less than 2, return
    if (joint_values.size() < 2)
    {
        RCLCPP_ERROR(LOGGER, "Failed to generate two states satisfying the task space constraint");
        return;
    }

    // get the first two states as start and goal states
    std::vector<double> start_joint_values_double;
    std::vector<float> start_joint_values_float;
    for(size_t i = 0; i < joint_values[0].size(); i++)
    {
        start_joint_values_double.push_back((double)joint_values[0][i]);
        start_joint_values_float.push_back(joint_values[0][i]);
    }
    std::vector<double> goal_joint_values_double;
    std::vector<float> goal_joint_values_float;
    for(size_t i = 0; i < joint_values[1].size(); i++)
    {
        goal_joint_values_double.push_back((double)joint_values[1][i]);
        goal_joint_values_float.push_back(joint_values[1][i]);
    }

    ///////////////////////////// Constrainted Motion Planning /////////////////////////////

    std::vector<std::vector<float>> start_joint_values_set;
    start_joint_values_set.push_back(start_joint_values_float);

    std::vector<std::vector<float>> goal_joint_values_set;
    goal_joint_values_set.push_back(goal_joint_values_float);

    // create the task
    CUDAMPLib::SingleArmTaskPtr task = std::make_shared<CUDAMPLib::SingleArmTask>(
        start_joint_values_set,
        goal_joint_values_set
    );

    // create the planner
    CUDAMPLib::RRGPtr planner = std::make_shared<CUDAMPLib::RRG>(single_arm_space);

    planner->setMaxTravelDistance(5.0);

    planner->setSampleAttemptsInEachIteration(100);

    // set the task
    planner->setMotionTask(task);

    CUDAMPLib::TimeoutTerminationPtr termination_condition = std::make_shared<CUDAMPLib::TimeoutTermination>(20.0);
    // CUDAMPLib::StepTerminationPtr termination_condition = std::make_shared<CUDAMPLib::StepTermination>(100);

    planner->solve(termination_condition);

    /************************** Debug **************************************/
    // extract the start and goal group states
    CUDAMPLib::BaseStatesPtr start_group_states;
    CUDAMPLib::BaseStatesPtr goal_group_states;
    planner->getStartAndGoalGroupStates(start_group_states, goal_group_states);

    // static_pointer_cast to SingleArmStates
    CUDAMPLib::SingleArmStatesPtr start_group_states_single_arm = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(start_group_states);
    CUDAMPLib::SingleArmStatesPtr goal_group_states_single_arm = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(goal_group_states);

    // create color
    std_msgs::msg::ColorRGBA color_start;
    color_start.r = 0.0;
    color_start.g = 1.0;
    color_start.b = 0.0;
    color_start.a = 0.4;

    // visualize the states
    visualization_msgs::msg::MarkerArray start_group_state_markers_combined;
    generate_state_markers(
        start_group_states_single_arm->getJointStatesHost(),
        joint_model_group,
        robot_state,
        "start_group",
        color_start,
        start_group_state_markers_combined,
        robot_info.getEndEffectorLinkNames()
    );

    // create color
    std_msgs::msg::ColorRGBA color_goal;
    color_goal.r = 1.0;
    color_goal.g = 0.0;
    color_goal.b = 0.0;
    color_goal.a = 0.4;

    visualization_msgs::msg::MarkerArray goal_group_state_markers_combined;
    generate_state_markers(
        goal_group_states_single_arm->getJointStatesHost(),
        joint_model_group,
        robot_state,
        "goal_group",
        color_goal,
        goal_group_state_markers_combined,
        robot_info.getEndEffectorLinkNames()
    );

    /************************** Debug **************************************/

    // Create a DisplayRobotState message
    moveit_msgs::msg::RobotState start_state_msg;
    robot_state->setJointGroupPositions(joint_model_group, start_joint_values_double);
    moveit::core::robotStateToRobotStateMsg(*robot_state, start_state_msg);
    moveit_msgs::msg::RobotState goal_state_msg;
    robot_state->setJointGroupPositions(joint_model_group, goal_joint_values_double);
    moveit::core::robotStateToRobotStateMsg(*robot_state, goal_state_msg);

    moveit_msgs::msg::DisplayRobotState start_display_robot_state;
    start_display_robot_state.state = start_state_msg;
    moveit_msgs::msg::DisplayRobotState goal_display_robot_state;
    goal_display_robot_state.state = goal_state_msg;

    moveit_msgs::msg::DisplayTrajectory display_trajectory;
    moveit_msgs::msg::RobotTrajectory robot_trajectory_msg;
    auto solution_robot_trajectory = robot_trajectory::RobotTrajectory(robot_model, joint_model_group);

    if (task->hasSolution())
    {
        // print "Task solved" in green color
        std::cout << "\033[1;32m" << "Task solved" << "\033[0m" << std::endl;

        std::vector<std::vector<float>> solution_path = task->getSolution();

        // generate robot trajectory msg
        for (size_t i = 0; i < solution_path.size(); i++)
        {
            // convert solution_path[i] to double vector
            std::vector<double> solution_path_i_double = std::vector<double>(solution_path[i].begin(), solution_path[i].end());
            robot_state->setJointGroupPositions(joint_model_group, solution_path_i_double);
            solution_robot_trajectory.addSuffixWayPoint(*robot_state, 1.0);

            if (i == 0)
            {
                moveit::core::robotStateToRobotStateMsg(*robot_state, start_state_msg);
            }

        }
        // Create a DisplayTrajectory message
        solution_robot_trajectory.getRobotTrajectoryMsg(robot_trajectory_msg);

        display_trajectory.trajectory_start = start_state_msg;
        display_trajectory.trajectory.push_back(robot_trajectory_msg);
    }
    else
    {
        // print "Task not solved" in red color
        std::cout << "\033[1;31m" << "Task not solved" << "\033[0m" << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////////////////

    // Create start/goal robot state publisher
    auto start_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("start_robot_state", 1);
    auto goal_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("goal_robot_state", 1);
    auto start_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("start_group_states", 1);
    auto goal_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("goal_group_states", 1);

    std::shared_ptr<rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>> display_publisher =
        node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);

    while (rclcpp::ok())
    {
        // Publish the message
        start_robot_state_publisher->publish(start_display_robot_state);
        goal_robot_state_publisher->publish(goal_display_robot_state);
        start_group_states_publisher->publish(start_group_state_markers_combined);
        goal_group_states_publisher->publish(goal_group_state_markers_combined);

        if (task->hasSolution())
        {
            display_publisher->publish(display_trajectory);
        }

        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    robot_state.reset();
}

void TEST_CHECK_CONSTRAINED_MOTION(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();

    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);
    
    // Prepare constraints
    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;
    
    // Create self collision constraint
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create task space constraint
    int task_link_index = -1;
    for (size_t i = 0; i < robot_info.getLinkNames().size(); i++)
    {
        if (robot_info.getLinkNames()[i] == robot_info.getEndEffectorLinkName())
        {
            task_link_index = i;
            break;
        }
    }
    if (task_link_index == -1)
    {
        RCLCPP_ERROR(LOGGER, "Failed to find the task link index");
        return;
    }

    std::vector<float> reference_frame = {0.9, 0.0, 0.7, 0.0, 0.0, 0.0};
    std::vector<float> tolerance = {1000, 1000, 1000, 0.0001, 0.0001, 10};
    CUDAMPLib::TaskSpaceConstraintPtr task_space_constraint = std::make_shared<CUDAMPLib::TaskSpaceConstraint>(
        "task_space_constraint",
        task_link_index,
        Eigen::Matrix4d::Identity(),
        reference_frame,
        tolerance
    );
    constraints.push_back(task_space_constraint);

    // Create boundary constraint
    CUDAMPLib::BoundaryConstraintPtr boundary_constraint = std::make_shared<CUDAMPLib::BoundaryConstraint>(
        "boundary_constraint",
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getActiveJointMap()
    );
    constraints.push_back(boundary_constraint);

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames(),
        0.02 // resolution
    );

    // generate start and goal states under the task space constraint by sampling
    CUDAMPLib::SingleArmStatesPtr start_single_arm_states = 
        std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(1000));
    CUDAMPLib::SingleArmStatesPtr goal_single_arm_states =
        std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(1000));

    size_t num_of_test_motions = 100;

    // check states
    start_single_arm_states->update();
    goal_single_arm_states->update();

    std::vector<bool> start_state_feasibility;
    std::vector<bool> goal_state_feasibility;
    single_arm_space->checkStates(start_single_arm_states, start_state_feasibility);
    single_arm_space->checkStates(goal_single_arm_states, goal_state_feasibility);

    // filter out the infeasible states
    start_single_arm_states->filterStates(start_state_feasibility);
    goal_single_arm_states->filterStates(goal_state_feasibility);

    if (start_single_arm_states->getJointStatesHost().size() < num_of_test_motions || goal_single_arm_states->getJointStatesHost().size() < num_of_test_motions)
    {
        // print in red
        std::cout << "\033[1;31m" << "Failed to generate enough start or goal states satisfying the task space constraint" << "\033[0m" << std::endl;
        return;
    }
    // create a vector of bool with size 5
    std::vector<bool> start_state_filter(start_single_arm_states->getNumOfStates(), false);
    std::vector<bool> goal_state_filter(goal_single_arm_states->getNumOfStates(), false);

    // only keep the first num_of_test_motions states
    for (size_t i = 0; i < num_of_test_motions; i++)
    {
        start_state_filter[i] = true;
        goal_state_filter[i] = true;
    }

    // filter out the states
    start_single_arm_states->filterStates(start_state_filter);
    goal_single_arm_states->filterStates(goal_state_filter);

    // print number of start and goal states
    std::cout << "Number of start states: " << start_single_arm_states->getNumOfStates() << std::endl;
    std::cout << "Number of goal states: " << goal_single_arm_states->getNumOfStates() << std::endl;

    // check constrained motions
    std::vector<bool> motion_feasibility;
    std::vector<float> motion_costs;
    auto start_time = std::chrono::high_resolution_clock::now();
    single_arm_space->checkConstrainedMotions(start_single_arm_states, goal_single_arm_states, motion_feasibility, motion_costs);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    // print in green
    std::cout << "\033[1;32m" << "Time taken by function: " << elapsed_time.count() << " seconds" << "\033[0m" << std::endl;

    // print the number of feasible motions
    int num_of_feasible_motions = 0;
    int first_valid_motion_index = -1;
    for (size_t i = 0; i < motion_feasibility.size(); i++)
    {
        if (motion_feasibility[i])
        {
            std::cout << "Motion " << i << " is feasible" << std::endl;
            num_of_feasible_motions++;
            if (first_valid_motion_index == -1)
            {
                first_valid_motion_index = i;
            }
        }
    }
    std::cout << "Number of feasible motions: " << num_of_feasible_motions << std::endl;

    if (num_of_feasible_motions == 0)
    {
        // print in red
        std::cout << "\033[1;31m" << "No feasible motions found" << "\033[0m" << std::endl;
        return;
    }

    // print the states of the first feasible motion
    std::vector<std::vector<float>> start_joint_values = start_single_arm_states->getJointStatesHost();

    std::vector<std::vector<float>> goal_joint_values = goal_single_arm_states->getJointStatesHost();

    // print the start and goal joint values
    std::cout << "Start joint values: ";
    for (size_t i = 0; i < start_joint_values[first_valid_motion_index].size(); i++)
    {
        std::cout << start_joint_values[first_valid_motion_index][i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Goal joint values: ";
    for (size_t i = 0; i < goal_joint_values[first_valid_motion_index].size(); i++)
    {
        std::cout << goal_joint_values[first_valid_motion_index][i] << " ";
    }
    std::cout << std::endl;

    std::vector<std::vector<float>> waypoints_joint_values = {start_joint_values[first_valid_motion_index], goal_joint_values[first_valid_motion_index]};

    // print the waypoints joint values
    std::cout << "Waypoints joint values: " << std::endl;
    for (size_t i = 0; i < waypoints_joint_values.size(); i++)
    {
        for (size_t j = 0; j < waypoints_joint_values[i].size(); j++)
        {
            std::cout << waypoints_joint_values[i][j] << " ";
        }
        std::cout << std::endl;
    }

    auto waypoints_states = single_arm_space->createStatesFromVector(waypoints_joint_values);
    auto fullpath = single_arm_space->getConstrainedPathFromWaypoints(waypoints_states);

    // cast the fullpath to SingleArmStates
    auto fullpath_single_arm_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(fullpath);

    ///////////////////////////// Visualize the path /////////////////////////////

    // create start state msg
    moveit_msgs::msg::RobotState start_state_msg;

    moveit_msgs::msg::DisplayTrajectory display_trajectory;
    moveit_msgs::msg::RobotTrajectory robot_trajectory_msg;
    auto solution_robot_trajectory = robot_trajectory::RobotTrajectory(robot_model, joint_model_group);

    std::vector<std::vector<float>> solution_path = fullpath_single_arm_states->getJointStatesHost();

    // generate robot trajectory msg
    for (size_t i = 0; i < solution_path.size(); i++)
    {
        // convert solution_path[i] to double vector
        std::vector<double> solution_path_i_double = std::vector<double>(solution_path[i].begin(), solution_path[i].end());
        robot_state->setJointGroupPositions(joint_model_group, solution_path_i_double);
        solution_robot_trajectory.addSuffixWayPoint(*robot_state, 1.0);

        if (i == 0)
        {
            // Create start msg
            moveit::core::robotStateToRobotStateMsg(*robot_state, start_state_msg);
        }
    }
    // Create a DisplayTrajectory message
    solution_robot_trajectory.getRobotTrajectoryMsg(robot_trajectory_msg);

    display_trajectory.trajectory_start = start_state_msg;
    display_trajectory.trajectory.push_back(robot_trajectory_msg);

    // Create a path publisher
    auto display_publisher = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);

    // Publish the message in a loop
    while (rclcpp::ok())
    {
        // Publish the message
        display_publisher->publish(display_trajectory);

        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}


void TEST_EVAL_MBM(const moveit::core::RobotModelPtr & robot_model, rclcpp::Node::SharedPtr node)
{
    /////////////////////////////// Setup ////////////////////////////////////////
    int task_index = 2;
    std::ostringstream oss;
    // Set the width to 5 and fill with '0'
    oss << std::setw(4) << std::setfill('0') << task_index;
    std::string task_index_str = oss.str();
   
    // Load problem dir
    std::string problem_dir = "/home/ros/problems/bookshelf_small_fetch";
    std::string robot_config_file = problem_dir + "/config.yaml";

    // load the robot config
    YAML::Node config = YAML::LoadFile(robot_config_file);

    // load the group name
    std::string planning_group = config["planning_group"].as<std::string>();
    std::cout << "Planning Group: " << planning_group << std::endl;

    //////////////////////////////// Load robot model ////////////////////////////////////////

    // Load robot model
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(planning_group);
    std::vector<std::string> joint_names = joint_model_group->getJointModelNames();
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);

    //////////////////////////// Load the scene objects ////////////////////////////

    // std::string scene_file = problem_dir + "/scene0001.yaml";
    std::string scene_file = problem_dir + "/scene" + task_index_str + ".yaml"; // with padding

    // load the scene
    std::vector<BoundingBox> boxes;
    std::vector<Cylinder> cylinders;
    std::vector<Sphere> spheres;

    if (loadSceneObjects(scene_file, boxes, cylinders, spheres)) {
        std::cout << "Loaded " << boxes.size() << " boxes" << std::endl;
        std::cout << "Loaded " << cylinders.size() << " cylinders" << std::endl;
        std::cout << "Loaded " << spheres.size() << " spheres" << std::endl;
    }

    // generate markers
    visualization_msgs::msg::MarkerArray box_markers = generateBoundingBoxesMarkers(boxes, node);
    visualization_msgs::msg::MarkerArray cylinder_markers = generateCylindersMarkers(cylinders, node);
    visualization_msgs::msg::MarkerArray sphere_markers = generateSpheresMarkers(spheres, node);

    // combine the markers
    visualization_msgs::msg::MarkerArray combined_markers;
    combined_markers.markers.insert(combined_markers.markers.end(), box_markers.markers.begin(), box_markers.markers.end());
    combined_markers.markers.insert(combined_markers.markers.end(), cylinder_markers.markers.begin(), cylinder_markers.markers.end());
    combined_markers.markers.insert(combined_markers.markers.end(), sphere_markers.markers.begin(), sphere_markers.markers.end());
    ///////////////////////////// Load start and goal ////////////////////////////////////////////////////////

    std::string request_file = problem_dir + "/request" + task_index_str + ".yaml";

    // load the start and goal
    std::vector<double> start_joint_values;
    std::vector<double> goal_joint_values;

    YAML::Node request_config = YAML::LoadFile(request_file);
    loadJointValues(request_config, joint_names, start_joint_values, goal_joint_values);

    // Print the joint values in the same order as joint_names.
    std::cout << "Task information:" << std::endl;
    for (size_t i = 0; i < joint_names.size(); ++i)
    {
        std::cout << "Joint: " << joint_names[i] 
                  << "  start: " << start_joint_values[i] 
                  << "  goal: " << goal_joint_values[i] << std::endl;
    }

    // use the start state is the default state
    std::map<std::string, double> default_map = loadStartStateJointState(request_config);
    std::map<std::string, float> default_map_float;
    // convert double to float
    for (const auto& pair : default_map)
    {
        default_map_float[pair.first] = static_cast<float>(pair.second);
    }

    // set robot_state with default map
    robot_state->setVariablePositions(default_map);

    // Prepare robot state for start and goal to visualize
    robot_state->setJointGroupPositions(joint_model_group, start_joint_values);
    moveit_msgs::msg::RobotState start_state_msg;
    moveit::core::robotStateToRobotStateMsg(*robot_state, start_state_msg);

    robot_state->setJointGroupPositions(joint_model_group, goal_joint_values);
    moveit_msgs::msg::RobotState goal_state_msg;
    moveit::core::robotStateToRobotStateMsg(*robot_state, goal_state_msg);

    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState start_display_robot_state;
    start_display_robot_state.state = start_state_msg;

    moveit_msgs::msg::DisplayRobotState goal_display_robot_state;
    goal_display_robot_state.state = goal_state_msg;

    ////////////////////////// Prepare robot info /////////////////////////////////////

    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, planning_group, collision_spheres_file_path, default_map_float);

    ////////////////////// Create space /////////////////////////////////////////////////////

    // Prepare constraints
    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    // Create self collision constraint
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    if (!boxes.empty())
    {
        std::vector<std::vector<float>> collision_box_pos;
        std::vector<std::vector<float>> collision_box_orientations;
        std::vector<std::vector<float>> collision_box_max;
        std::vector<std::vector<float>> collision_box_min;
        CuboidToVectors(boxes, collision_box_pos, collision_box_orientations, collision_box_max, collision_box_min);
        CUDAMPLib::EnvConstraintCuboidPtr environment_collision_constraint_cuboid = std::make_shared<CUDAMPLib::EnvConstraintCuboid>(
            "environment_collision_constraint_cuboid",
            collision_box_pos,
            collision_box_orientations,
            collision_box_max,
            collision_box_min
        );
        constraints.push_back(environment_collision_constraint_cuboid);
    }

    if (!cylinders.empty())
    {
        std::vector<std::vector<float>> collision_cylinder_pos;
        std::vector<std::vector<float>> collision_cylinder_orientations;
        std::vector<float> collision_cylinder_radius;
        std::vector<float> collision_cylinder_height;
        CylinderToVectors(cylinders, collision_cylinder_pos, collision_cylinder_orientations, collision_cylinder_radius, collision_cylinder_height);
        CUDAMPLib::EnvConstraintCylinderPtr environment_collision_constraint_cylinder = std::make_shared<CUDAMPLib::EnvConstraintCylinder>(
            "environment_collision_constraint_cylinder",
            collision_cylinder_pos,
            collision_cylinder_orientations,
            collision_cylinder_radius,
            collision_cylinder_height
        );
        constraints.push_back(environment_collision_constraint_cylinder);
    }

    if (!spheres.empty())
    {
        std::vector<std::vector<float>> collision_sphere_pos;
        std::vector<float> collision_sphere_radius;
        SphereToVectors(spheres, collision_sphere_pos, collision_sphere_radius);
        CUDAMPLib::EnvConstraintSpherePtr environment_collision_constraint_sphere = std::make_shared<CUDAMPLib::EnvConstraintSphere>(
            "environment_collision_constraint_sphere",
            collision_sphere_pos,
            collision_sphere_radius
        );
        constraints.push_back(environment_collision_constraint_sphere);
    }

    // Create space
    CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
        robot_info.getDimension(),
        constraints,
        robot_info.getJointTypes(),
        robot_info.getJointPoses(),
        robot_info.getJointAxes(),
        robot_info.getLinkMaps(),
        robot_info.getCollisionSpheresMap(),
        robot_info.getCollisionSpheresPos(),
        robot_info.getCollisionSpheresRadius(),
        robot_info.getActiveJointMap(),
        robot_info.getLowerBounds(),
        robot_info.getUpperBounds(),
        robot_info.getDefaultJointValues(),
        robot_info.getLinkNames(),
        0.05 // resolution
    );

    ///////////////////////////////////////// create the task /////////////////////////////////////

    std::vector<std::vector<float>> start_joint_values_set;
    // convert start_joint_values to float
    std::vector<float> start_joint_values_float;
    for(size_t i = 0; i < start_joint_values.size(); i++)
    {
        start_joint_values_float.push_back((float)start_joint_values[i]);
    }
    start_joint_values_set.push_back(start_joint_values_float);
    std::vector<std::vector<float>> goal_joint_values_set;
    // convert goal_joint_values to float
    std::vector<float> goal_joint_values_float;
    for(size_t i = 0; i < goal_joint_values.size(); i++)
    {
        goal_joint_values_float.push_back((float)goal_joint_values[i]);
    }
    goal_joint_values_set.push_back(goal_joint_values_float);

    // create the task
    CUDAMPLib::SingleArmTaskPtr task = std::make_shared<CUDAMPLib::SingleArmTask>(
        start_joint_values_set,
        goal_joint_values_set
    );

    // create the planner

    CUDAMPLib::RRGPtr planner = std::make_shared<CUDAMPLib::RRG>(single_arm_space);
    planner->setMaxTravelDistance(1.0);
    planner->setSampleAttemptsInEachIteration(100);
    // set the task
    planner->setMotionTask(task);
    CUDAMPLib::TimeoutTerminationPtr termination_condition = std::make_shared<CUDAMPLib::TimeoutTermination>(20.0);

    auto start_time = std::chrono::high_resolution_clock::now();
    planner->solve(termination_condition);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    // print in green
    std::cout << "\033[1;32m" << "Time taken by function: " << elapsed_time.count() << " seconds" << "\033[0m" << std::endl;

    /************************** Visualize both start and goal group **************************************/

    // extract the start and goal group states
    CUDAMPLib::BaseStatesPtr start_group_states;
    CUDAMPLib::BaseStatesPtr goal_group_states;
    planner->getStartAndGoalGroupStates(start_group_states, goal_group_states);

    // static_pointer_cast to SingleArmStates
    CUDAMPLib::SingleArmStatesPtr start_group_states_single_arm = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(start_group_states);
    CUDAMPLib::SingleArmStatesPtr goal_group_states_single_arm = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(goal_group_states);

    // create color for start group
    std_msgs::msg::ColorRGBA color_start;
    color_start.r = 0.0;
    color_start.g = 1.0;
    color_start.b = 0.0;
    color_start.a = 0.4;

    // visualize the start group states
    visualization_msgs::msg::MarkerArray start_group_state_markers_combined;
    generate_state_markers(
        start_group_states_single_arm->getJointStatesHost(),
        joint_model_group,
        robot_state,
        "start_group",
        color_start,
        start_group_state_markers_combined,
        robot_info.getEndEffectorLinkNames()
    );

    // create color for goal group
    std_msgs::msg::ColorRGBA color_goal;
    color_goal.r = 1.0;
    color_goal.g = 0.0;
    color_goal.b = 0.0;
    color_goal.a = 0.4;

    visualization_msgs::msg::MarkerArray goal_group_state_markers_combined;
    generate_state_markers(
        goal_group_states_single_arm->getJointStatesHost(),
        joint_model_group,
        robot_state,
        "goal_group",
        color_goal,
        goal_group_state_markers_combined,
        robot_info.getEndEffectorLinkNames()
    );

    ////////////////////////////////////// Visualize path //////////////////////////////////////

    moveit_msgs::msg::DisplayTrajectory display_trajectory;
    moveit_msgs::msg::RobotTrajectory robot_trajectory_msg;
    auto solution_robot_trajectory = robot_trajectory::RobotTrajectory(robot_model, joint_model_group);

    if (task->hasSolution())
    {
        // print "Task solved" in green color
        std::cout << "\033[1;32m" << "Task solved" << "\033[0m" << std::endl;

        std::vector<std::vector<float>> solution_path = task->getSolution();

        // generate robot trajectory msg
        for (size_t i = 0; i < solution_path.size(); i++)
        {
            // convert solution_path[i] to double vector
            std::vector<double> solution_path_i_double = std::vector<double>(solution_path[i].begin(), solution_path[i].end());
            robot_state->setJointGroupPositions(joint_model_group, solution_path_i_double);
            solution_robot_trajectory.addSuffixWayPoint(*robot_state, 1.0);
        }
        // Create a DisplayTrajectory message
        solution_robot_trajectory.getRobotTrajectoryMsg(robot_trajectory_msg);

        display_trajectory.trajectory_start = start_state_msg;
        display_trajectory.trajectory.push_back(robot_trajectory_msg);
    }
    else
    {
        // print "Task not solved" in red color
        std::cout << "\033[1;31m" << "Task not solved" << "\033[0m" << std::endl;

        // print the failure reason
        std::cout << "Failure reason: " << task->getFailureReason() << std::endl;
    }

    //////////////////////////////////////// Publish the markers ////////////////////////////////////////

    auto marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("scene_objects", 1);
    auto start_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("start_robot_state", 1);
    auto goal_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("goal_robot_state", 1);
    auto start_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("start_group_states", 1);
    auto goal_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("goal_group_states", 1);
    auto display_publisher = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);

    // Publish the message in a loop
    while (rclcpp::ok())
    {
        // Publish the message
        marker_publisher->publish(combined_markers);
        start_robot_state_publisher->publish(start_display_robot_state);
        goal_robot_state_publisher->publish(goal_display_robot_state);
        start_group_states_publisher->publish(start_group_state_markers_combined);
        goal_group_states_publisher->publish(goal_group_state_markers_combined);
        if (task->hasSolution())
        {
            display_publisher->publish(display_trajectory);
        }

        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

}

void VIS_RESULT_MBM(const moveit::core::RobotModelPtr & robot_model, rclcpp::Node::SharedPtr node)
{
    //////////////////////////// Load Result /////////////////////////////////////
    std::string result_file = "/home/ros/pRRTC/build/path.txt";

    std::ifstream
    infile(result_file);
    std::string line;
    std::vector<std::vector<double>> result_path;
    // get the first line which is the task index
    std::getline(infile, line);
    // get the task index as int
    int task_index = std::stoi(line);
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<double> joint_values;
        double value;
        while (iss >> value)
        {
            joint_values.push_back(value);
        }
        result_path.push_back(joint_values);
    }

    // print the result path
    std::cout << "Result path: " << std::endl;
    for (size_t i = 0; i < result_path.size(); i++)
    {
        for (size_t j = 0; j < result_path[i].size(); j++)
        {
            std::cout << result_path[i][j] << " ";
        }
        std::cout << std::endl;
    }

    /////////////////////////////// Setup ////////////////////////////////////////
    // int task_index = 33;
    std::ostringstream oss;
    // Set the width to 5 and fill with '0'
    oss << std::setw(4) << std::setfill('0') << task_index;
    std::string task_index_str = oss.str();
   
    // Load problem dir
    std::string problem_dir = "/home/ros/problems/bookshelf_small_fetch";
    std::string robot_config_file = problem_dir + "/config.yaml";

    // load the robot config
    YAML::Node config = YAML::LoadFile(robot_config_file);

    // load the group name
    std::string planning_group = config["planning_group"].as<std::string>();
    std::cout << "Planning Group: " << planning_group << std::endl;

    //////////////////////////////// Load robot model ////////////////////////////////////////

    // Load robot model
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(planning_group);
    std::vector<std::string> joint_names = joint_model_group->getJointModelNames();
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);

    //////////////////////////// Load the scene objects ////////////////////////////

    // std::string scene_file = problem_dir + "/scene0001.yaml";
    std::string scene_file = problem_dir + "/scene" + task_index_str + ".yaml"; // with padding

    // load the scene
    std::vector<BoundingBox> boxes;
    std::vector<Cylinder> cylinders;
    std::vector<Sphere> spheres;

    if (loadSceneObjects(scene_file, boxes, cylinders, spheres)) {
        std::cout << "Loaded " << boxes.size() << " boxes" << std::endl;
        std::cout << "Loaded " << cylinders.size() << " cylinders" << std::endl;
        std::cout << "Loaded " << spheres.size() << " spheres" << std::endl;
    }

    // generate markers
    visualization_msgs::msg::MarkerArray box_markers = generateBoundingBoxesMarkers(boxes, node);
    visualization_msgs::msg::MarkerArray cylinder_markers = generateCylindersMarkers(cylinders, node);
    visualization_msgs::msg::MarkerArray sphere_markers = generateSpheresMarkers(spheres, node);

    // combine the markers
    visualization_msgs::msg::MarkerArray combined_markers;
    combined_markers.markers.insert(combined_markers.markers.end(), box_markers.markers.begin(), box_markers.markers.end());
    combined_markers.markers.insert(combined_markers.markers.end(), cylinder_markers.markers.begin(), cylinder_markers.markers.end());
    combined_markers.markers.insert(combined_markers.markers.end(), sphere_markers.markers.begin(), sphere_markers.markers.end());
    ///////////////////////////// Load start and goal ////////////////////////////////////////////////////////

    std::string request_file = problem_dir + "/request" + task_index_str + ".yaml";

    // load the start and goal
    std::vector<double> start_joint_values;
    std::vector<double> goal_joint_values;

    YAML::Node request_config = YAML::LoadFile(request_file);
    loadJointValues(request_config, joint_names, start_joint_values, goal_joint_values);

    // Print the joint values in the same order as joint_names.
    std::cout << "Task information:" << std::endl;
    for (size_t i = 0; i < joint_names.size(); ++i)
    {
        std::cout << "Joint: " << joint_names[i] 
                  << "  start: " << start_joint_values[i] 
                  << "  goal: " << goal_joint_values[i] << std::endl;
    }

    // use the start state is the default state
    std::map<std::string, double> default_map = loadStartStateJointState(request_config);
    std::map<std::string, float> default_map_float;
    // convert double to float
    for (const auto& pair : default_map)
    {
        default_map_float[pair.first] = static_cast<float>(pair.second);
    }

    // set robot_state with default map
    robot_state->setVariablePositions(default_map);

    // Prepare robot state for start and goal to visualize
    robot_state->setJointGroupPositions(joint_model_group, start_joint_values);
    moveit_msgs::msg::RobotState start_state_msg;
    moveit::core::robotStateToRobotStateMsg(*robot_state, start_state_msg);

    robot_state->setJointGroupPositions(joint_model_group, goal_joint_values);
    moveit_msgs::msg::RobotState goal_state_msg;
    moveit::core::robotStateToRobotStateMsg(*robot_state, goal_state_msg);

    // Create a DisplayRobotState message
    moveit_msgs::msg::DisplayRobotState start_display_robot_state;
    start_display_robot_state.state = start_state_msg;

    moveit_msgs::msg::DisplayRobotState goal_display_robot_state;
    goal_display_robot_state.state = goal_state_msg;



    // ////////////////////////////////////// Visualize path //////////////////////////////////////

    moveit_msgs::msg::DisplayTrajectory display_trajectory;
    moveit_msgs::msg::RobotTrajectory robot_trajectory_msg;
    auto solution_robot_trajectory = robot_trajectory::RobotTrajectory(robot_model, joint_model_group);

    for (size_t i = 0; i < result_path.size(); i++)
    {
        robot_state->setJointGroupPositions(joint_model_group, result_path[i]);
        solution_robot_trajectory.addSuffixWayPoint(*robot_state, 1.0);
    }
    // Create a DisplayTrajectory message
    solution_robot_trajectory.getRobotTrajectoryMsg(robot_trajectory_msg);

    display_trajectory.trajectory_start = start_state_msg;
    display_trajectory.trajectory.push_back(robot_trajectory_msg);

    // //////////////////////////////////////// Publish the markers ////////////////////////////////////////

    auto marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("scene_objects", 1);
    auto start_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("start_robot_state", 1);
    auto goal_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("goal_robot_state", 1);
    auto display_publisher = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);
    auto cylinder_markers_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("cylinder_markers", 1);
    auto sphere_markers_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("sphere_markers", 1);
    auto box_markers_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("box_markers", 1);

    // Publish the message in a loop
    while (rclcpp::ok())
    {
        // Publish the message
        marker_publisher->publish(combined_markers);
        start_robot_state_publisher->publish(start_display_robot_state);
        goal_robot_state_publisher->publish(goal_display_robot_state);
        cylinder_markers_publisher->publish(cylinder_markers);
        sphere_markers_publisher->publish(sphere_markers);
        box_markers_publisher->publish(box_markers);

        display_publisher->publish(display_trajectory);

        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

}

void TEST_OBSTACLES(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    // get robot state
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();

    // find the region where the obstacles should not be placed
    std::vector<BoundingBox> unmoveable_bounding_boxes_of_robot = getUnmoveableBoundingBoxes(robot_model, group_name, 0.05);

    // generate collision spheres
    std::vector<Sphere> collision_spheres;
    genSphereObstacles(40, 0.08, 0.06, unmoveable_bounding_boxes_of_robot, collision_spheres);

    // generate collision cuboids
    std::vector<BoundingBox> collision_cuboids;
    genCuboidObstacles(40, 0.08, 0.06, unmoveable_bounding_boxes_of_robot, collision_cuboids);

    // generate collisino cylinders
    std::vector<Cylinder> collision_cylinders;
    genCylinderObstacles(40, 0.08, 0.06, 0.1, 0.05, unmoveable_bounding_boxes_of_robot, collision_cylinders);


    // Generate markers for the obstacles
    visualization_msgs::msg::MarkerArray sphere_obstacle_marker_array = generateSpheresMarkers(collision_spheres, node);
    visualization_msgs::msg::MarkerArray cuboid_obstacle_marker_array = generateBoundingBoxesMarkers(collision_cuboids, node);
    visualization_msgs::msg::MarkerArray cylinder_obstacle_marker_array = generateCylindersMarkers(collision_cylinders, node);

    // Create a obstacle MarkerArray publisher
    auto sphere_obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);
    auto cuboid_obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_cuboids", 1);
    auto cylinder_obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_cylinders", 1);

    // convert those bounding boxes to markers and publish them
    visualization_msgs::msg::MarkerArray marker_array;
    int id_counter = 0;

    for (const auto& b : unmoveable_bounding_boxes_of_robot)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";  // Replace with appropriate frame if needed
        marker.header.stamp = rclcpp::Clock().now();
        marker.ns = "unmoveable_bounding_boxes";
        marker.id = id_counter++;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Size of the box
        marker.scale.x = b.x_max - b.x_min;
        marker.scale.y = b.y_max - b.y_min;
        marker.scale.z = b.z_max - b.z_min;

        // Position and orientation
        marker.pose.position.x = b.x;
        marker.pose.position.y = b.y;
        marker.pose.position.z = b.z;

        tf2::Quaternion q;
        q.setRPY(b.roll, b.pitch, b.yaw);
        marker.pose.orientation = tf2::toMsg(q);

        // Color (e.g., red with alpha)
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.5;

        // Lifetime and other properties
        marker_array.markers.push_back(marker);
    }

    // visualize the robot state
    moveit_msgs::msg::RobotState robot_state_msg;
    moveit::core::robotStateToRobotStateMsg(*robot_state, robot_state_msg);

    moveit_msgs::msg::DisplayRobotState display_robot_state;
    display_robot_state.state = robot_state_msg;

    // Create a robot state publisher
    auto robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("goal_robot_state", 1);
    auto unmoveable_bounding_boxes_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("unmoveable_region", 1);

    // Publish the message in a loop
    while (rclcpp::ok())
    {
        // Publish the message
        unmoveable_bounding_boxes_publisher->publish(marker_array);

        robot_state_publisher->publish(display_robot_state);

        sphere_obstacle_marker_publisher->publish(sphere_obstacle_marker_array);
        cuboid_obstacle_marker_publisher->publish(cuboid_obstacle_marker_array);
        cylinder_obstacle_marker_publisher->publish(cylinder_obstacle_marker_array);

        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main(int argc, char** argv)
{
    // const std::string GROUP_NAME = "arm"; // Fetch
    // const std::string GROUP_NAME = "fr3_arm"; // franka

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto cuda_test_node = rclcpp::Node::make_shared("cuda_test_node", node_options);

    // get group name
    std::string GROUP_NAME;
    cuda_test_node->get_parameter("group_name", GROUP_NAME);
    std::cout << "Group name: " << GROUP_NAME << std::endl;

    // print out the node name
    RCLCPP_INFO(LOGGER, "Node name: %s", cuda_test_node->get_name());

    // Create a robot model
    robot_model_loader::RobotModelLoaderPtr robot_model_loader(
      new robot_model_loader::RobotModelLoader(cuda_test_node, "robot_description"));
    const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader->getModel();
    if (kinematic_model == nullptr)
    {
        RCLCPP_ERROR(cuda_test_node->get_logger(), "Failed to load robot model");
        return 1;
    }

    // =========================================================================================

    // // print collision_spheres_file_path from ros parameter server
    // std::string collision_spheres_file_path;
    // cuda_test_node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    // RCLCPP_INFO(cuda_test_node->get_logger(), "collision_spheres_file_path: %s", collision_spheres_file_path.c_str());

    // TEST_OBSTACLES(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_FORWARD(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_JACOBIAN(kinematic_model, GROUP_NAME, cuda_test_node);

    // EVAL_FORWARD(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_CONSTRAINT_PROJECT(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_TASK_WITH_GOAL_REGION(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_COLLISION(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_COLLISION_AND_VIS(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_NEAREST_NEIGHBOR(kinematic_model, GROUP_NAME, cuda_test_node);

    TEST_Planner(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_OMPL(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_CONSTRAINED_MOTION_PLANNING(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_CHECK_CONSTRAINED_MOTION(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_FILTER_STATES(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_EVAL_MBM(kinematic_model, cuda_test_node);

    // VIS_RESULT_MBM(kinematic_model, cuda_test_node);

    // list ros parameters
    // RCLCPP_INFO(cuda_test_node->get_logger(), "List all parameters");
    // auto parameters = cuda_test_node->list_parameters({""}, 5);
    // for (const auto& parameter : parameters.names)
    // {
    //     RCLCPP_INFO(cuda_test_node->get_logger(), "Parameter name: %s", parameter.c_str());
    // }

    // stop the node
    rclcpp::shutdown();

    return 0;
}
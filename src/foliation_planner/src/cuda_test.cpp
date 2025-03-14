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
#include <CUDAMPLib/constraints/EnvConstraint.h>
#include <CUDAMPLib/constraints/SelfCollisionConstraint.h>
#include <CUDAMPLib/constraints/TaskSpaceConstraint.h>
#include <CUDAMPLib/constraints/BoundaryConstraint.h>
#include <CUDAMPLib/tasks/SingleArmTask.h>
#include <CUDAMPLib/planners/RRG.h>
#include <CUDAMPLib/termination/StepTermination.h>
#include <CUDAMPLib/termination/TimeoutTermination.h>

// ompl include
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>

#include "foliation_planner/robot_info.hpp"

// include for time
#include <chrono>

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

void prepare_obstacles(std::vector<std::vector<float>> & balls_pos, std::vector<float> & ball_radius)
{
    float obstacle_spheres_radius = 0.06;
    int num_of_obstacle_spheres = 40;
    for (int i = 0; i < num_of_obstacle_spheres; i++)
    {
        float x = 0.3 * ((float)rand() / RAND_MAX) + 0.3;
        float y = 2.0 * 0.5 * ((float)rand() / RAND_MAX) - 0.5;
        float z = 1.0 * ((float)rand() / RAND_MAX) + 0.5;
        balls_pos.push_back({x, y, z});
        ball_radius.push_back(obstacle_spheres_radius);
    }
}

void generate_state_markers(
    const std::vector<std::vector<float>> & group_joint_values,
    const moveit::core::JointModelGroup* joint_model_group,
    moveit::core::RobotStatePtr robot_state,
    const std::string & group_ns,
    const std_msgs::msg::ColorRGBA color,
    visualization_msgs::msg::MarkerArray & robot_marker_array
)
{
    std::vector<visualization_msgs::msg::MarkerArray> group_state_markers;

    std::vector<std::string> display_links_names = joint_model_group->getLinkModelNames();
    // add end effector link by hard code
    display_links_names.push_back(std::string("gripper_link"));
    display_links_names.push_back(std::string("r_gripper_finger_link"));
    display_links_names.push_back(std::string("l_gripper_finger_link"));

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

void TEST_JACOBIAN(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    // create moveit robot state
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
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

    std::string check_link_name = "wrist_roll_link";

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
        // Eigen::Isometry3d end_effector_link_pose = robot_state->getGlobalLinkTransform("wrist_roll_link");
        Eigen::Isometry3d end_effector_link_pose = robot_state->getGlobalLinkTransform(check_link_name);
        std::cout << "End effector pose " << i << " using moveit: " << std::endl;
        std::cout << "position: " << end_effector_link_pose.translation().transpose() << std::endl;
        // std::cout << end_effector_link_pose.rotation() << std::endl;
        // print it as quaternion
        Eigen::Quaterniond q_moveit(end_effector_link_pose.rotation());
        std::cout << "quaternion: " << q_moveit.w() << " " << q_moveit.x() << " " << q_moveit.y() << " " << q_moveit.z() << std::endl;

        // compute Jacobian
        Eigen::MatrixXd jacobian;
        // robot_state->getJacobian(joint_model_group, robot_state->getLinkModel("wrist_roll_link"), Eigen::Vector3d(0, 0, 0), jacobian);
        robot_state->getJacobian(joint_model_group, robot_state->getLinkModel(check_link_name), Eigen::Vector3d(0, 0, 0), jacobian);
        std::cout << "Jacobian: " << std::endl;
        std::cout << jacobian << std::endl;
    }
}

/**
    Use moveit to compute the forward kinematics
 */
void TEST_FORWARD(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    // create moveit robot state
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
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

    std::string check_link_name = "wrist_roll_link";

    std::vector<std::vector<float>> moveit_positions;
    std::vector<std::vector<float>> moveit_orientations;

    int test_config_num = 1000;
    std::vector<std::vector<float>> joint_values_set;
    for (int i = 0; i < test_config_num; i++)
    {
        // use moveit to randomly sample joint values
        std::vector<double> joint_values_double;
        robot_state->setToRandomPositions(joint_model_group);
        robot_state->copyJointGroupPositions(joint_model_group, joint_values_double);
        std::vector<float> joint_values_float;
        for (size_t j = 0; j < joint_values_double.size(); j++)
        {
            joint_values_float.push_back((float)joint_values_double[j]);
        }
        joint_values_set.push_back(joint_values_float);

        // store the end effector pose
        robot_state->update();
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
            std::cout << "\033[1;31m" << "Error in position" << "\033[0m" << std::endl;
            continue;
        }

        if (fabs((float)(q.w()) - moveit_orientations[i][0]) > 0.01 ||
            fabs((float)(q.x()) - moveit_orientations[i][1]) > 0.01 ||
            fabs((float)(q.y()) - moveit_orientations[i][2]) > 0.01 ||
            fabs((float)(q.z()) - moveit_orientations[i][3]) > 0.01)
        {
            // print in red
            std::cout << "\033[1;31m" << "Error in orientation" << "\033[0m" << std::endl;
            continue;
        }

        // print in green
        std::cout << "\033[1;32m" << "Same poses" << "\033[0m" << std::endl;
    }

}

void EVAL_FORWARD(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    // create moveit robot state
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
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

void TEST_COLLISION(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    
    // // create obstacles manually
    // balls_pos.push_back({0.4, 0.0, 1.4});
    // ball_radius.push_back(0.2);

    // create obstacles randomly
    prepare_obstacles(balls_pos, ball_radius);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::EnvConstraintPtr env_constraint = std::make_shared<CUDAMPLib::EnvConstraint>(
        "obstacle_constraint",
        balls_pos,
        ball_radius
    );
    constraints.push_back(env_constraint);

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
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

    // dumpy update
    single_arm_states_1->oldUpdate();

    auto start_time_update = std::chrono::high_resolution_clock::now();
    single_arm_states_2->update();
    auto end_time_update = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time_update = end_time_update - start_time_update;
    // print in green color
    printf("\033[1;32m" "Time taken by update: %f seconds" "\033[0m \n", elapsed_time_update.count());

    auto start_time_old_update = std::chrono::high_resolution_clock::now();
    single_arm_states_3->oldUpdate();
    auto end_time_old_update = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time_old_update = end_time_old_update - start_time_old_update;
    // print in green color
    printf("\033[1;32m" "Time taken by old update: %f seconds" "\033[0m \n", elapsed_time_old_update.count());

    // check states
    single_arm_space->oldCheckStates(single_arm_states_1); // dummy check

    auto start_time_check_states = std::chrono::high_resolution_clock::now();
    single_arm_space->checkStates(single_arm_states_2);
    auto end_time_check_states = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time_check_states = end_time_check_states - start_time_check_states;
    // print in green color
    printf("\033[1;32m" "Time taken by checkStates: %f seconds" "\033[0m \n", elapsed_time_check_states.count());

    auto start_time_old_check_states = std::chrono::high_resolution_clock::now();
    single_arm_space->oldCheckStates(single_arm_states_3);
    auto end_time_old_check_states = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time_old_check_states = end_time_old_check_states - start_time_old_check_states;
    // print in green color
    printf("\033[1;32m" "Time taken by oldCheckStates: %f seconds" "\033[0m \n", elapsed_time_old_check_states.count());

    // // check motions
    // std::vector<bool> motion_feasibility;
    // std::vector<float> motion_costs;

    // auto start_time_check_motions = std::chrono::high_resolution_clock::now();
    // single_arm_space->checkMotions(single_arm_states_1, single_arm_states_2, motion_feasibility, motion_costs);
    // auto end_time_check_motions = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed_time_check_motions = end_time_check_motions - start_time_check_motions;
    // // print in green color
    // printf("\033[1;32m" "Time taken by checkMotions: %f seconds" "\033[0m \n", elapsed_time_check_motions.count());

    // std::vector<bool> motion_feasibility_1;
    // std::vector<float> motion_costs_1;

    // start_time_check_motions = std::chrono::high_resolution_clock::now();
    // single_arm_space->oldCheckMotions(single_arm_states_1, single_arm_states_2, motion_feasibility_1, motion_costs_1);
    // end_time_check_motions = std::chrono::high_resolution_clock::now();
    // elapsed_time_check_motions = end_time_check_motions - start_time_check_motions;
    // // print in green color
    // printf("\033[1;32m" "Time taken by oldCheckMotions: %f seconds" "\033[0m \n", elapsed_time_check_motions.count());

    // start_time_check_motions = std::chrono::high_resolution_clock::now();
    // single_arm_space->checkMotions(single_arm_states_2, single_arm_states_3, motion_feasibility, motion_costs);
    // end_time_check_motions = std::chrono::high_resolution_clock::now();
    // elapsed_time_check_motions = end_time_check_motions - start_time_check_motions;
    // // print in green color
    // printf("\033[1;32m" "Time taken by second checkMotions: %f seconds" "\033[0m \n", elapsed_time_check_motions.count());
}

void TEST_CONSTRAINT_PROJECT(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;

    // create obstacles randomly
    prepare_obstacles(balls_pos, ball_radius);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    // CUDAMPLib::EnvConstraintPtr env_constraint = std::make_shared<CUDAMPLib::EnvConstraint>(
    //     "obstacle_constraint",
    //     balls_pos,
    //     ball_radius
    // );
    // constraints.push_back(env_constraint);

    int task_link_index = -1;
    for (size_t i = 0; i < robot_info.getLinkNames().size(); i++)
    {
        if (robot_info.getLinkNames()[i] == "wrist_roll_link")
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
    generate_state_markers(states_joint_values, joint_model_group, robot_state, "sample_group", color_sample, sample_group_state_markers);
    
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

void TEST_TASK_WITH_GOAL_REGION(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    int task_link_index = -1;
    for (size_t i = 0; i < robot_info.getLinkNames().size(); i++)
    {
        if (robot_info.getLinkNames()[i] == "wrist_roll_link")
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
    generate_state_markers(states_joint_values, joint_model_group, robot_state, "sample_group", color_sample, sample_group_state_markers);

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
void TEST_FILTER_STATES(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
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
    std::cout << "\033[1;32m" << "Time taken by old filter state function: " << elapsed_time_filter.count() << " seconds" << "\033[0m" << std::endl;
}

/**
    Create a CUDAMPLib::SingleArmSpace and sample a set of states.
    Then, we will check the feasibility of the states and visualize the collision spheres in rviz.
 */
void TEST_COLLISION_AND_VIS(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    prepare_obstacles(balls_pos, ball_radius);

    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

    CUDAMPLib::EnvConstraintPtr env_constraint = std::make_shared<CUDAMPLib::EnvConstraint>(
        "obstacle_constraint",
        balls_pos,
        ball_radius
    );
    constraints.push_back(env_constraint);

    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
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
    // add end effector link by hard code
    display_links_names.push_back(std::string("gripper_link"));
    display_links_names.push_back(std::string("r_gripper_finger_link"));
    display_links_names.push_back(std::string("l_gripper_finger_link"));

    std::vector<visualization_msgs::msg::MarkerArray> sample_group_state_markers;

    std::vector<std::vector<float>> states_joint_values = sampled_states->getJointStatesFullHost();
    for (size_t i = 0; i < states_joint_values.size(); i++)
    {
        // for (size_t j = 0; j < states_joint_values[i].size(); j++)
        // {
        //     std::cout << states_joint_values[i][j] << " ";
        // }

        // if (state_feasibility[i])
        // {
        //     std::cout << " feasible" << std::endl;
        // }
        // else
        // {
        //     std::cout << " infeasible" << std::endl;
        // }

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
        }
        else
        {
            color.r = 1.0;
            color.g = 0.0;
            color.b = 0.0;
            color.a = 0.4;
        }
        const std::string sample_group_ns = "sampled_group";
        robot_state->getRobotMarkers(robot_marker, display_links_names, color, sample_group_ns, rclcpp::Duration::from_seconds(0));
        sample_group_state_markers.push_back(robot_marker);
    }

    visualization_msgs::msg::MarkerArray sample_group_state_markers_combined;
    for (size_t i = 0; i < sample_group_state_markers.size(); i++)
    {
        sample_group_state_markers_combined.markers.insert(
            sample_group_state_markers_combined.markers.end(), 
            sample_group_state_markers[i].markers.begin(), sample_group_state_markers[i].markers.end());
    }

    // update the id
    for (size_t i = 0; i < sample_group_state_markers_combined.markers.size(); i++)
    {
        sample_group_state_markers_combined.markers[i].id = i;
    }

    std::vector<std::vector<std::vector<float>>> self_collision_spheres_pos =  sampled_states->getSelfCollisionSpheresPosInBaseLinkHost();

    std::vector<std::vector<float>> collision_spheres_pos_of_selected_config = self_collision_spheres_pos[0];

    /*======================================= prepare publishers ==================================================================== */

    // Create marker publisher
    auto self_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("self_collision_spheres", 1);
    auto obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);
    auto sample_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("sample_group_states", 1);
    // Create a self MarkerArray message
    visualization_msgs::msg::MarkerArray robot_collision_spheres_marker_array = generate_self_collision_markers(
        collision_spheres_pos_of_selected_config,
        robot_info.getCollisionSpheresRadius(),
        node
    );
    // Create a obstacle MarkerArray message
    visualization_msgs::msg::MarkerArray obstacle_collision_spheres_marker_array = generate_obstacles_markers(balls_pos, ball_radius, node);

    // use loop to publish the trajectory
    while (rclcpp::ok())
    {
        // Publish the message
        self_marker_publisher->publish(robot_collision_spheres_marker_array);
        obstacle_marker_publisher->publish(obstacle_collision_spheres_marker_array);
        sample_group_states_publisher->publish(sample_group_state_markers_combined);
        
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

void TEST_Planner(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    // Prepare obstacle constraint
    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    prepare_obstacles(balls_pos, ball_radius);

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

    // Prepare constraints
    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;
    // Create obstacle constraint
    CUDAMPLib::EnvConstraintPtr env_constraint = std::make_shared<CUDAMPLib::EnvConstraint>(
        "obstacle_constraint",
        balls_pos,
        ball_radius
    );
    constraints.push_back(env_constraint);
    // Create self collision constraint
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
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
        0.02 // resolution
    );

    std::vector<std::vector<float>> start_joint_values_set;
    start_joint_values_set.push_back(start_joint_values);

    std::vector<std::vector<float>> goal_joint_values_set;
    goal_joint_values_set.push_back(goal_joint_values);
    // create the task
    CUDAMPLib::SingleArmTaskPtr task = std::make_shared<CUDAMPLib::SingleArmTask>(
        start_joint_values_set,
        goal_joint_values_set
    );

    // create the planner
    CUDAMPLib::RRGPtr planner = std::make_shared<CUDAMPLib::RRG>(single_arm_space);

    // set parameters
    planner->setK(1);

    planner->setMaxTravelDistance(5.0);

    planner->setSampleAttemptsInEachIteration(100);
    
    // set the task
    planner->setMotionTask(task);

    // create termination condition
    // CUDAMPLib::StepTerminationPtr termination_condition = std::make_shared<CUDAMPLib::StepTermination>(10);
    CUDAMPLib::TimeoutTerminationPtr termination_condition = std::make_shared<CUDAMPLib::TimeoutTermination>(10.0);

    // solve the task
    // record the time
    auto start_time = std::chrono::high_resolution_clock::now();
    planner->solve(termination_condition);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    // print in green
    std::cout << "\033[1;32m" << "Time taken by function: " << elapsed_time.count() << " seconds" << "\033[0m" << std::endl;

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
        start_group_state_markers_combined
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
        goal_group_state_markers_combined
    );

    /************************** Debug **************************************/

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
    }

    /************************************* prepare publishers ******************************************* */

    // Create a start robot state publisher
    auto start_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("start_robot_state", 1);
    // Create a goal robot state publisher
    auto goal_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("goal_robot_state", 1);
    auto obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);
    std::shared_ptr<rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>> display_publisher =
        node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);

    auto start_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("start_group_states", 1);
    auto goal_group_states_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("goal_group_states", 1);

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

    // clear the robot state
    robot_state.reset();
}

void TEST_OMPL(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    // Prepare obstacle constraint
    std::vector<std::vector<float>> balls_pos;
    std::vector<float> ball_radius;
    prepare_obstacles(balls_pos, ball_radius);

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
void TEST_CONSTRAINED_MOTION_PLANNING(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);
    
    // Prepare constraints
    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;
    
    // Create self collision constraint
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create task space constraint
    int task_link_index = -1;
    for (size_t i = 0; i < robot_info.getLinkNames().size(); i++)
    {
        if (robot_info.getLinkNames()[i] == "wrist_roll_link")
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

    planner->setK(1);

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
        start_group_state_markers_combined
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
        goal_group_state_markers_combined
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

void TEST_CHECK_CONSTRAINED_MOTION(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();

    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);
    
    // Prepare constraints
    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;
    
    // Create self collision constraint
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    // Create task space constraint
    int task_link_index = -1;
    for (size_t i = 0; i < robot_info.getLinkNames().size(); i++)
    {
        if (robot_info.getLinkNames()[i] == "wrist_roll_link")
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

int main(int argc, char** argv)
{
    const std::string GROUP_NAME = "arm";

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto cuda_test_node = rclcpp::Node::make_shared("cuda_test_node", node_options);

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

    // TEST_FORWARD(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_JACOBIAN(kinematic_model, GROUP_NAME, cuda_test_node);

    EVAL_FORWARD(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_CONSTRAINT_PROJECT(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_TASK_WITH_GOAL_REGION(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_COLLISION(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_COLLISION_AND_VIS(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_Planner(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_OMPL(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_CONSTRAINED_MOTION_PLANNING(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_CHECK_CONSTRAINED_MOTION(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_FILTER_STATES(kinematic_model, GROUP_NAME, cuda_test_node);

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
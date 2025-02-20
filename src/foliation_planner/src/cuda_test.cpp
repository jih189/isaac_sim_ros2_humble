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
    int num_of_obstacle_spheres = 30;
    for (int i = 0; i < num_of_obstacle_spheres; i++)
    {
        float x = 0.3 * ((float)rand() / RAND_MAX) + 0.3;
        float y = 2.0 * 0.5 * ((float)rand() / RAND_MAX) - 0.5;
        float z = 1.0 * ((float)rand() / RAND_MAX) + 0.5;
        balls_pos.push_back({x, y, z});
        ball_radius.push_back(obstacle_spheres_radius);
    }
}

void TEST_FORWARD(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
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

    // set a test joint values
    std::vector<float> joint_values_1 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> joint_values_2 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5};
    std::vector<float> joint_values_3 = {0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0};
    std::vector<std::vector<float>> joint_values_set;
    joint_values_set.push_back(joint_values_1);
    joint_values_set.push_back(joint_values_2);
    joint_values_set.push_back(joint_values_3);

    // create states based on the joint values
    auto states = single_arm_space->createStatesFromVector(joint_values_set);
    states->update();

    // statistic_cast_pointer_cast to SingleArmStates
    CUDAMPLib::SingleArmStatesPtr single_arm_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(states);

    std::vector<Eigen::Isometry3d> end_effector_link_poses_in_base_link = single_arm_states->getLinkPoseInBaseLinkHost("wrist_roll_link");

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
    }
}

/**
    Create a CUDAMPLib::SingleArmSpace and sample a set of states.
    Then, we will check the feasibility of the states and visualize the collision spheres in rviz.
 */
void TEST_CUDAMPLib(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, bool debug = false)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path, debug);

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
    CUDAMPLib::SingleArmStatesPtr sampled_states = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(single_arm_space->sample(10));
    sampled_states->update();

    std::vector<bool> state_feasibility;

    // check states
    single_arm_space->checkStates(sampled_states, state_feasibility);

    std::vector<std::vector<float>> states_joint_values = sampled_states->getJointStatesHost();
    for (size_t i = 0; i < states_joint_values.size(); i++)
    {
        for (size_t j = 0; j < states_joint_values[i].size(); j++)
        {
            std::cout << states_joint_values[i][j] << " ";
        }

        if (state_feasibility[i])
        {
            std::cout << " feasible" << std::endl;
        }
        else
        {
            std::cout << " infeasible" << std::endl;
        }
    }

    std::vector<std::vector<std::vector<float>>> self_collision_spheres_pos =  sampled_states->getSelfCollisionSpheresPosInBaseLinkHost();

    std::vector<std::vector<float>> collision_spheres_pos_of_selected_config = self_collision_spheres_pos[0];

    /*======================================= prepare publishers ==================================================================== */

    // Create marker publisher
    auto self_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("self_collision_spheres", 1);
    auto obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);
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
        
        rclcpp::spin_some(node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
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

    // set the task
    planner->setMotionTask(task);

    // create termination condition
    // CUDAMPLib::StepTerminationPtr termination_condition = std::make_shared<CUDAMPLib::StepTermination>(10);
    CUDAMPLib::TimeoutTerminationPtr termination_condition = std::make_shared<CUDAMPLib::TimeoutTermination>(1.0);

    // solve the task
    planner->solve(termination_condition);

    /************************** Debug **************************************/
    // extract the start and goal group states
    CUDAMPLib::BaseStatesPtr start_group_states;
    CUDAMPLib::BaseStatesPtr goal_group_states;
    planner->getStartAndGoalGroupStates(start_group_states, goal_group_states);

    // static_pointer_cast to SingleArmStates
    CUDAMPLib::SingleArmStatesPtr start_group_states_single_arm = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(start_group_states);
    CUDAMPLib::SingleArmStatesPtr goal_group_states_single_arm = std::static_pointer_cast<CUDAMPLib::SingleArmStates>(goal_group_states);

    // get the joint values
    std::vector<std::vector<float>> start_group_joint_values = start_group_states_single_arm->getJointStatesHost();
    std::vector<std::vector<float>> goal_group_joint_values = goal_group_states_single_arm->getJointStatesHost();

    std::vector<visualization_msgs::msg::MarkerArray> start_group_state_markers;
    std::vector<visualization_msgs::msg::MarkerArray> goal_group_state_markers;

    std::vector<std::string> display_links_names = joint_model_group->getLinkModelNames();
    // add end effector link by hard code
    display_links_names.push_back(std::string("gripper_link"));
    display_links_names.push_back(std::string("r_gripper_finger_link"));
    display_links_names.push_back(std::string("l_gripper_finger_link"));

    // print start and goal group joint values
    for (size_t i = 0; i < start_group_joint_values.size(); i++)
    {
        std::vector<double> start_group_joint_values_i_double;
        for (size_t j = 0; j < start_group_joint_values[i].size(); j++)
        {
            // print only active joints
            if (robot_info.getActiveJointMap()[j])
            {
                start_group_joint_values_i_double.push_back((double)start_group_joint_values[i][j]);
            }
        }

        robot_state->setJointGroupPositions(joint_model_group, start_group_joint_values_i_double);
        robot_state->update();
        visualization_msgs::msg::MarkerArray robot_marker;
        // color
        std_msgs::msg::ColorRGBA color;
        color.r = 1.0;
        color.g = 0.0;
        color.b = 0.0;
        color.a = 0.4;
        const std::string start_group_ns = "start_group";
        robot_state->getRobotMarkers(robot_marker, display_links_names, color, start_group_ns, rclcpp::Duration::from_seconds(0));
        start_group_state_markers.push_back(robot_marker);
    }

    for (size_t i = 0; i < goal_group_joint_values.size(); i++)
    {
        std::vector<double> goal_group_joint_values_i_double;
        for (size_t j = 0; j < goal_group_joint_values[i].size(); j++)
        {
            if (robot_info.getActiveJointMap()[j])
            {
                goal_group_joint_values_i_double.push_back((double)goal_group_joint_values[i][j]);
            }
        }

        robot_state->setJointGroupPositions(joint_model_group, goal_group_joint_values_i_double);
        robot_state->update();
        visualization_msgs::msg::MarkerArray robot_marker;
        // color
        std_msgs::msg::ColorRGBA color;
        color.r = 0.0;
        color.g = 1.0;
        color.b = 0.0;
        color.a = 0.4;
        const std::string goal_group_ns = "goal_group";
        robot_state->getRobotMarkers(robot_marker, display_links_names, color, goal_group_ns, rclcpp::Duration::from_seconds(0));

        goal_group_state_markers.push_back(robot_marker);
    }

    // conbine  start and goal group state markers
    visualization_msgs::msg::MarkerArray start_group_state_markers_combined;
    visualization_msgs::msg::MarkerArray goal_group_state_markers_combined;
    for (size_t i = 0; i < start_group_state_markers.size(); i++)
    {
        start_group_state_markers_combined.markers.insert(start_group_state_markers_combined.markers.end(), start_group_state_markers[i].markers.begin(), start_group_state_markers[i].markers.end());
    }

    // update the id
    for (size_t i = 0; i < start_group_state_markers_combined.markers.size(); i++)
    {
        start_group_state_markers_combined.markers[i].id = i;
    }

    for (size_t i = 0; i < goal_group_state_markers.size(); i++)
    {
        goal_group_state_markers_combined.markers.insert(goal_group_state_markers_combined.markers.end(), goal_group_state_markers[i].markers.begin(), goal_group_state_markers[i].markers.end());
    }

    // update the id
    for (size_t i = 0; i < goal_group_state_markers_combined.markers.size(); i++)
    {
        goal_group_state_markers_combined.markers[i].id = i;
    }

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

    TEST_FORWARD(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_CUDAMPLib(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_Planner(kinematic_model, GROUP_NAME, cuda_test_node);

    // TEST_OMPL(kinematic_model, GROUP_NAME, cuda_test_node);

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
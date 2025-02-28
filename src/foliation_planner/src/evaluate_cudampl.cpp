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

#include "foliation_planner/robot_info.hpp"

#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>

#include <chrono>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("cudampl Evaluation");

struct MotionPlanningTask
{
    std::vector<std::vector<float>> obstacle_pos;
    std::vector<float> radius;
    std::vector<float> start_joint_values;
    std::vector<float> goal_joint_values;
};

std::vector<MotionPlanningTask> loadMotionPlanningTasks(const std::string & task_dir_path, rclcpp::Node::SharedPtr node)
{
    // loading motion tasks
    std::vector<MotionPlanningTask> tasks;

    // find the metadata file
    std::string metadata_path = task_dir_path + "/metadata.txt";
    std::ifstream metadata(metadata_path);

    if (!metadata.is_open())
    {
        RCLCPP_ERROR(node->get_logger(), "Failed to open metadata file");
        return tasks;
    }

    std::string line;
    while (std::getline(metadata, line))
    {
        // load the task file
        std::string task_file_path = task_dir_path + "/" + line;
        std::ifstream task_file(task_file_path);

        if (!task_file.is_open())
        {
            RCLCPP_ERROR(node->get_logger(), "Failed to open task file");
            continue;
        }

        YAML::Node task_node = YAML::LoadFile(task_file_path);
        MotionPlanningTask task;

        // // get the start joint values
        std::vector<float> start_joint_values = task_node["Start joint values"].as<std::vector<float>>();
        task.start_joint_values = start_joint_values;

        // get the goal joint values
        std::vector<float> goal_joint_values = task_node["Goal joint values"].as<std::vector<float>>();
        task.goal_joint_values = goal_joint_values;

        // get the obstacles
        auto obstacles = task_node["Obstacles"];

        for (size_t i = 0; i < obstacles.size(); i++)
        {
            std::vector<float> obstacle_pos = obstacles[i]["Position"].as<std::vector<float>>();
            float obstacle_radius = obstacles[i]["Radius"].as<float>();
            // print out the obstacle position
            // RCLCPP_INFO(LOGGER, "Obstacle %zu position: [%f, %f, %f] with radius %f ", i, obstacle_pos[0], obstacle_pos[1], obstacle_pos[2], obstacle_radius);
            task.obstacle_pos.push_back(obstacle_pos);
            task.radius.push_back(obstacle_radius);
        }

        tasks.push_back(task);
    }
    return tasks;
}

moveit_msgs::msg::DisplayRobotState getDisplayRobotState(
    const moveit::core::RobotStatePtr & robot_state, 
    const moveit::core::JointModelGroup * joint_model_group, 
    const std::vector<float> & joint_values)
{
    std::vector<double> joint_values_double;
    for (size_t i = 0; i < joint_values.size(); i++)
    {
        joint_values_double.push_back((double)joint_values[i]);
    }
    robot_state->setJointGroupPositions(joint_model_group, joint_values_double);
    robot_state->update();
    moveit_msgs::msg::DisplayRobotState robot_state_msg;
    moveit::core::robotStateToRobotStateMsg(*robot_state, robot_state_msg.state);
    return robot_state_msg;
}

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

void Eval_Planner(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, std::vector<MotionPlanningTask> & tasks)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    // // Create publishers
    // auto start_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("start_robot_state", 1);
    // auto goal_robot_state_publisher = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("goal_robot_state", 1);
    // auto obstacle_marker_publisher = node->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_collision_spheres", 1);

    // Prepare constraints
    std::vector<CUDAMPLib::BaseConstraintPtr> constraints;
    
    // Create self collision constraint
    CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
        "self_collision_constraint",
        robot_info.getSelfCollisionEnabledMap()
    );
    constraints.push_back(self_collision_constraint);

    long int total_time = 0;
    long int total_solved = 0;
    long int total_unsolved = 0;

    // print out the tasks
    for (size_t i = 0; i < tasks.size(); i++)
    {
        RCLCPP_INFO(LOGGER, "Task %zu", i);

        // if constraints contains "obstacle_constraint", then remove it
        for (size_t j = 0; j < constraints.size(); j++)
        {
            if (constraints[j]->getName() == "obstacle_constraint")
            {
                constraints.erase(constraints.begin() + j);
                break;
            }
        }
        // Create obstacle constraint
        CUDAMPLib::EnvConstraintPtr env_constraint = std::make_shared<CUDAMPLib::EnvConstraint>(
            "obstacle_constraint",
            tasks[i].obstacle_pos,
            tasks[i].radius
        );
        constraints.push_back(env_constraint);

        // // print number of constraints
        // RCLCPP_INFO(LOGGER, "Number of constraints: %zu", constraints.size());

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
        start_joint_values_set.push_back(tasks[i].start_joint_values);

        std::vector<std::vector<float>> goal_joint_values_set;
        goal_joint_values_set.push_back(tasks[i].goal_joint_values);

        // create the task
        CUDAMPLib::SingleArmTaskPtr problem_task = std::make_shared<CUDAMPLib::SingleArmTask>(
            start_joint_values_set,
            goal_joint_values_set
        );

        // create the planner
        CUDAMPLib::RRGPtr planner = std::make_shared<CUDAMPLib::RRG>(single_arm_space);

        // // set parameters
        planner->setK(1);

        planner->setMaxTravelDistance(5.0);

        planner->setSampleAttemptsInEachIteration(100);

        // set the task
        planner->setMotionTask(problem_task, false);

        // record the start time
        auto start_time = std::chrono::high_resolution_clock::now();

        // solve the task
        CUDAMPLib::TimeoutTerminationPtr termination_condition = std::make_shared<CUDAMPLib::TimeoutTermination>(10.0);
        planner->solve(termination_condition);

        // record the end time
        auto end_time = std::chrono::high_resolution_clock::now();
        // calculate the time taken
        auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        RCLCPP_INFO(LOGGER, "Time taken: %ld ms", time_taken);

        // check if the task is solved
        if (problem_task->hasSolution())
        {
            RCLCPP_INFO(LOGGER, "Task %zu is solved", i);
            total_solved++;
            total_time += time_taken;
        }
        else
        {
            if (problem_task->getFailureReason() == "MeetTerminationCondition")
            {
                RCLCPP_INFO(LOGGER, "Task %zu is not solved due to timeout", i);
                RCLCPP_INFO(LOGGER, "Failure reason: %s", problem_task->getFailureReason().c_str());
                total_unsolved++;
            }
        }

        planner.reset();
        single_arm_space.reset();
        problem_task.reset();
    }

    // print out the average time
    RCLCPP_INFO(LOGGER, "Average time: %ld ms", total_time / total_solved);
    // print success rate
    float success_rate = (float)total_solved / (float)(total_solved + total_unsolved);
    RCLCPP_INFO(LOGGER, "Success rate: %f", success_rate);
}


int main(int argc, char** argv)
{
    // ============================ parameters =================================== //
    std::string task_dir_path = "/home/motion_planning_tasks";
    const std::string GROUP_NAME = "arm";
    // =========================================================================== //

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto motion_planning_evaluation_node = rclcpp::Node::make_shared("motion_planning_evaluation_node", node_options);

    // print out the node name
    RCLCPP_INFO(LOGGER, "Node name: %s", motion_planning_evaluation_node->get_name());

    // Create a robot model
    robot_model_loader::RobotModelLoaderPtr robot_model_loader(
      new robot_model_loader::RobotModelLoader(motion_planning_evaluation_node, "robot_description"));
    const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader->getModel();
    if (kinematic_model == nullptr)
    {
        RCLCPP_ERROR(motion_planning_evaluation_node->get_logger(), "Failed to load robot model");
        return 1;
    }

    std::vector<MotionPlanningTask> tasks = loadMotionPlanningTasks(task_dir_path, motion_planning_evaluation_node);
    
    // print out the number of tasks
    RCLCPP_INFO(LOGGER, "Number of tasks: %zu", tasks.size());

    Eval_Planner(kinematic_model, GROUP_NAME, motion_planning_evaluation_node, tasks);

    // stop the node
    rclcpp::shutdown();

    return 0;
}
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
#include <CUDAMPLib/multiply.h>
#include <CUDAMPLib/kinematics.h>
#include <CUDAMPLib/cost.h>
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
        else
        {
            RCLCPP_INFO(LOGGER, "Loading task file: %s", task_file_path.c_str());
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
    
    // // print out the number of tasks
    // RCLCPP_INFO(LOGGER, "Number of tasks: %zu", tasks.size());

    // // print out the tasks
    // for (size_t i = 0; i < tasks.size(); i++)
    // {
    //     RCLCPP_INFO(LOGGER, "Task %zu", i);

    //     // print out the start joint values
    //     RCLCPP_INFO(LOGGER, "Start joint values: ");
    //     for (size_t j = 0; j < tasks[i].start_joint_values.size(); j++)
    //     {
    //         RCLCPP_INFO(LOGGER, "%f ", tasks[i].start_joint_values[j]);
    //     }

    //     // print out the goal joint values
    //     RCLCPP_INFO(LOGGER, "Goal joint values: ");
    //     for (size_t j = 0; j < tasks[i].goal_joint_values.size(); j++)
    //     {
    //         RCLCPP_INFO(LOGGER, "%f ", tasks[i].goal_joint_values[j]);
    //     }

    //     for (size_t j = 0; j < tasks[i].obstacle_pos.size(); j++)
    //     {
    //         RCLCPP_INFO(LOGGER, "Obstacle %zu position: [%f, %f, %f] with radius %f ", j, tasks[i].obstacle_pos[j][0], tasks[i].obstacle_pos[j][1], tasks[i].obstacle_pos[j][2], tasks[i].radius[j]);
    //     }
    // }

    // stop the node
    rclcpp::shutdown();

    return 0;
}
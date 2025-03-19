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

#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("task generator");

struct MotionPlanningTask
{
    std::vector<std::vector<float>> obstacle_pos;
    std::vector<float> radius;
    std::vector<float> start_joint_values;
    std::vector<float> goal_joint_values;
};

// void prepare_obstacles(std::vector<std::vector<float>> & balls_pos, std::vector<float> & ball_radius)
// {
//     float obstacle_spheres_radius = 0.06;
//     int num_of_obstacle_spheres = 20;
//     for (int i = 0; i < num_of_obstacle_spheres; i++)
//     {
//         float x = 0.3 * ((float)rand() / RAND_MAX) + 0.3;
//         float y = 2.0 * 0.5 * ((float)rand() / RAND_MAX) - 0.5;
//         float z = 1.0 * ((float)rand() / RAND_MAX) + 0.5;
//         balls_pos.push_back({x, y, z});
//         ball_radius.push_back(obstacle_spheres_radius);
//     }
// }

void prepare_obstacles(std::vector<std::vector<float>> & balls_pos, std::vector<float> & ball_radius, const std::string & group_name)
{
    if (group_name == "arm"){
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
    else if (group_name == "fr3_arm")
    {
        float obstacle_spheres_radius = 0.06;
        int num_of_obstacle_spheres = 40;
        for (int i = 0; i < num_of_obstacle_spheres; i++)
        {
            float x = 1.4 * ((float)rand() / RAND_MAX) - 0.7;
            float y = 1.4 * ((float)rand() / RAND_MAX) - 0.7;
            float z = 1.0 * ((float)rand() / RAND_MAX) + 0.0;

            if (
                x > -0.2 && x < 0.2 &&
                y > -0.2 && y < 0.2 &&
                z > 0.0 && z < 0.6
            )
                continue;

            balls_pos.push_back({x, y, z});
            ball_radius.push_back(obstacle_spheres_radius);
        }
    }
    else
    {
        std::cout << "Group name is not supported!" << std::endl;
    }
}


std::vector<MotionPlanningTask> generateMotionPlanningTasks(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, int num_tasks)
{
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);
    
    std::vector<MotionPlanningTask> tasks;
    for(int i = 0; i < num_tasks; i++)
    {
        MotionPlanningTask task;
        // generate random obstacles
        prepare_obstacles(task.obstacle_pos, task.radius, group_name);

        // generate start and goal states
        std::vector<float> start_joint_values;
        std::vector<float> goal_joint_values;

        // generate random start state
        robot_state->setToRandomPositions();
        robot_state->update();

        std::vector<double> start_joint_values_double;
        robot_state->copyJointGroupPositions(joint_model_group, start_joint_values_double);
        for (size_t j = 0; j < start_joint_values_double.size(); j++)
        {
            start_joint_values.push_back((float)start_joint_values_double[j]);
        }
        task.start_joint_values = start_joint_values;

        // generate random goal state
        robot_state->setToRandomPositions();
        robot_state->update();

        std::vector<double> goal_joint_values_double;
        robot_state->copyJointGroupPositions(joint_model_group, goal_joint_values_double);
        for (size_t j = 0; j < goal_joint_values_double.size(); j++)
        {
            goal_joint_values.push_back((float)goal_joint_values_double[j]);
        }

        task.goal_joint_values = goal_joint_values;

        tasks.push_back(task);
    }
    robot_state.reset();

    return tasks;
}

bool isTaskValid(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, const MotionPlanningTask & task)
{
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    // create planning scene
    auto world = std::make_shared<collision_detection::World>();
    auto planning_scene = std::make_shared<planning_scene::PlanningScene>(robot_model, world);

    // add those balls to the planning scene
    for (size_t i = 0; i < task.obstacle_pos.size(); i++)
    {
        Eigen::Isometry3d sphere_pose = Eigen::Isometry3d::Identity();
        sphere_pose.translation() = Eigen::Vector3d(task.obstacle_pos[i][0], task.obstacle_pos[i][1], task.obstacle_pos[i][2]);
        planning_scene->getWorldNonConst()->addToObject("obstacle_" + std::to_string(i), shapes::ShapeConstPtr(new shapes::Sphere(task.radius[i])), sphere_pose);
    }

    std::vector<double> start_state_double;
    std::vector<double> goal_state_double;

    for (size_t i = 0; i < task.start_joint_values.size(); i++)
    {
        start_state_double.push_back((double)task.start_joint_values[i]);
        goal_state_double.push_back((double)task.goal_joint_values[i]);
    }
    robot_state->setJointGroupPositions(joint_model_group, start_state_double);
    robot_state->update();

    // check if the start state is valid and self-collision free
    if (!planning_scene->isStateValid(*robot_state, group_name))
    {
        return false;
    }

    robot_state->setJointGroupPositions(joint_model_group, goal_state_double);
    robot_state->update();

    // check if the goal state is valid and self-collision free
    if (!planning_scene->isStateValid(*robot_state, group_name))
    {
        return false;
    }

    return true;
}


int main(int argc, char** argv)
{
    
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto motion_planning_evaluation_node = rclcpp::Node::make_shared("motion_planning_evaluation_node", node_options);

    // ============================ parameters =================================== //
    // std::string task_dir_path = "/home/motion_planning_tasks";
    int num_tasks = 100;
    std::string GROUP_NAME;
    motion_planning_evaluation_node->get_parameter("group_name", GROUP_NAME);
    std::string task_dir_path = "/home/motion_planning_tasks/" + GROUP_NAME;
    // =========================================================================== //

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

    // generate motion planning tasks
    std::vector<MotionPlanningTask> tasks = generateMotionPlanningTasks(kinematic_model, GROUP_NAME, num_tasks);
    
    // check if the directory exists, if so, then delete it
    if (std::filesystem::exists(task_dir_path))
    {
        // delete the directory
        std::string delete_command = "rm -rf " + task_dir_path;
        int r1 = system(delete_command.c_str());
        if (r1 != 0)
        {
            RCLCPP_ERROR(motion_planning_evaluation_node->get_logger(), "Failed to delete directory");
            return 1;
        }
    }
    // create the directory
    std::string create_command = "mkdir -p " + task_dir_path;
    int r2 = system(create_command.c_str());
    if (r2 != 0)
    {
        RCLCPP_ERROR(motion_planning_evaluation_node->get_logger(), "Failed to create directory");
        return 1;
    }

    // create a file containing all task files' names in the directory
    std::string metadata_path = task_dir_path + "/metadata.txt";
    std::ofstream metadata(metadata_path);

    for (size_t i = 0; i < tasks.size(); i++)
    {
        auto task = tasks[i];

        // check if the task is valid
        if (!isTaskValid(kinematic_model, GROUP_NAME, task))
        {
            RCLCPP_WARN(LOGGER, "Task %zu is invalid", i);
            continue;
        }

        // create a file for each task
        std::string task_file_path = task_dir_path + "/task_" + std::to_string(i) + ".yaml";
        std::ofstream task_file(task_file_path);

        // save the task to the file
        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "Start joint values" << YAML::Value << task.start_joint_values;
        out << YAML::Key << "Goal joint values" << YAML::Value << task.goal_joint_values;
        out << YAML::Key << "Obstacles" << YAML::Value;
        out << YAML::BeginSeq;
        for (size_t j = 0; j < task.obstacle_pos.size(); j++)
        {
            out << YAML::BeginMap;
            out << YAML::Key << "Position" << YAML::Value << task.obstacle_pos[j];
            out << YAML::Key << "Radius" << YAML::Value << task.radius[j];
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;
        out << YAML::EndMap;

        // write to file
        metadata << "task_" + std::to_string(i) + ".yaml" + "\n";
        
        // write to file
        task_file.write(out.c_str(), out.size());
        
        // close the file
        task_file.close();
    }

    // close the file
    metadata.close();

    // stop the node
    rclcpp::shutdown();

    return 0;
}
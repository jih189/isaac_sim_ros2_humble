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

// ompl include
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>

#include "foliation_planner/robot_info.hpp"

namespace ob = ompl::base;
namespace og = ompl::geometric;

#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>

#include <chrono>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("ompl Evaluation");

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

void Eval_Planner(const moveit::core::RobotModelPtr & robot_model, const std::string & group_name, rclcpp::Node::SharedPtr node, std::vector<MotionPlanningTask> & tasks)
{
    std::string collision_spheres_file_path;
    node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);
    RobotInfo robot_info(robot_model, group_name, collision_spheres_file_path);

    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
    // set robot state to default state
    robot_state->setToDefaultValues();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    // set group dimension
    int dim = robot_model->getJointModelGroup(group_name)->getActiveJointModels().size();

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

    // create ompl states for start and goal
    ompl::base::ScopedState<> start_state(ompl::base::StateSpacePtr(new ompl::base::RealVectorStateSpace(dim)));
    ompl::base::ScopedState<> goal_state(ompl::base::StateSpacePtr(new ompl::base::RealVectorStateSpace(dim)));

    long int total_time = 0;
    long int total_solved = 0;
    long int total_unsolved = 0;

    // print out the tasks
    for (size_t i = 0; i < tasks.size(); i++)
    {
        // create planning scene
        auto world = std::make_shared<collision_detection::World>();
        auto planning_scene = std::make_shared<planning_scene::PlanningScene>(robot_model, world);

        // add those balls to the planning scene
        for (size_t j = 0; j < tasks[i].obstacle_pos.size(); j++)
        {
            Eigen::Isometry3d sphere_pose = Eigen::Isometry3d::Identity();
            sphere_pose.translation() = Eigen::Vector3d(tasks[i].obstacle_pos[j][0], tasks[i].obstacle_pos[j][1], tasks[i].obstacle_pos[j][2]);
            planning_scene->getWorldNonConst()->addToObject("obstacle_" + std::to_string(j), shapes::ShapeConstPtr(new shapes::Sphere(tasks[i].radius[j])), sphere_pose);
        }

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

        si->setup();

        // set start and goal state
        for (int j = 0; j < dim; j++)
        {
            start_state[j] = tasks[i].start_joint_values[j];
            goal_state[j] = tasks[i].goal_joint_values[j];
        }

        // set problem definition
        ompl::base::ProblemDefinitionPtr pdef(new ompl::base::ProblemDefinition(si));
        pdef->setStartAndGoalStates(start_state, goal_state);

        // create planner
        auto planner(std::make_shared<og::RRTConnect>(si));
        planner->setProblemDefinition(pdef);
        planner->setup();

        // record the start time
        auto start_time = std::chrono::high_resolution_clock::now();

        // solve the problem
        ompl::base::PlannerStatus solved = planner->ob::Planner::solve(10.0);

        // record the end time
        auto end_time = std::chrono::high_resolution_clock::now();

        // calculate the time taken
        auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // print out the time taken
        RCLCPP_INFO(LOGGER, "Time taken: %ld ms", time_taken);

        if (solved)
        {
            RCLCPP_INFO(LOGGER, "Task %zu is solved", i);
            total_solved++;
            total_time += time_taken;
        }
        else
        {
            RCLCPP_INFO(LOGGER, "Task %zu is not solved", i);
            total_unsolved++;
        }
        
        // reset planner
        world.reset();
        planning_scene.reset();
        si.reset();
        planner.reset();
        pdef.reset();
    }

    // print out the average time
    RCLCPP_INFO(LOGGER, "Average time: %ld ms", total_time / total_solved);
    float success_rate = (float)total_solved / (float)(total_solved + total_unsolved);
    RCLCPP_INFO(LOGGER, "Success rate: %f", success_rate);

    robot_state.reset();
    
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
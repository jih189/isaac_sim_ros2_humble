#include <rclcpp/rclcpp.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include "moveit/planning_interface/planning_interface.h"
#include "moveit/robot_state/conversions.h"
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
// #include <moveit/robot_state/joint_model_group.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("lerp_example");

int main(int argc, char** argv)
{
    const std::string GROUP_NAME = "arm";

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto lerp_example_node = rclcpp::Node::make_shared("lerp_example_node", node_options);

    // print out the node name
    RCLCPP_INFO(LOGGER, "Node name: %s", lerp_example_node->get_name());

    // rclcpp::executors::SingleThreadedExecutor executor;
    // executor.add_node(lerp_example_node);
    // std::thread([&executor]() { executor.spin(); }).detach();
        
    // Create a robot model
    robot_model_loader::RobotModelLoaderPtr robot_model_loader(
      new robot_model_loader::RobotModelLoader(lerp_example_node, "robot_description"));
    const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader->getModel();
    if (kinematic_model == nullptr)
    {
        RCLCPP_ERROR(lerp_example_node->get_logger(), "Failed to load robot model");
        return 1;
    }

    // Using the RobotModelLoader, we can construct a planning scene monitor that
    // will create a planning scene, monitors planning scene diffs, and apply the diffs to it's
    // internal planning scene. We then call startSceneMonitor, startWorldGeometryMonitor and
    // startStateMonitor to fully initialize the planning scene monitor
    planning_scene_monitor::PlanningSceneMonitorPtr psm(
        new planning_scene_monitor::PlanningSceneMonitor(lerp_example_node, robot_model_loader));

    /* listen for planning scene messages on topic /XXX and apply them to the internal planning scene
                       the internal planning scene accordingly */
    psm->startSceneMonitor();
    /* listens to changes of world geometry, collision objects, and (optionally) octomaps
                                    world geometry, collision objects and optionally octomaps */
    psm->startWorldGeometryMonitor();
    /* listen to joint state updates as well as changes in attached collision objects
                            and update the internal planning scene accordingly*/
    psm->startStateMonitor();
    
    // Create a robot state
    moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(kinematic_model);
    
    // Get the joint model group
    const moveit::core::JointModelGroup* joint_model_group = robot_state->getJointModelGroup(GROUP_NAME);

    // Get the joint names
    const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();

    // Print the joint names
    for (const auto& joint_name : joint_names)
    {
        RCLCPP_INFO(lerp_example_node->get_logger(), "Joint name: %s", joint_name.c_str());
    }

    // Print End Effector Link
    RCLCPP_INFO(lerp_example_node->get_logger(), "End Effector Link: %s", joint_model_group->getLinkModelNames().back().c_str());

    // // Set the planner plugin
    // lerp_example_node->set_parameter(rclcpp::Parameter("planning_plugin", "lerp_interface/LerpPlanner"));

    // Create pipeline, where third argumentn is parameter namespace
    planning_pipeline::PlanningPipelinePtr planning_pipeline(
      new planning_pipeline::PlanningPipeline(kinematic_model, lerp_example_node, "lerp"));

    planning_interface::MotionPlanRequest req;
    planning_interface::MotionPlanResponse res;

    // Set the start and goal joint values
    std::vector<double> start_joint_vals = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> goal_joint_vals = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2};

    req.group_name = GROUP_NAME;

    // Get the start state
    robot_state->setJointGroupPositions(joint_model_group, start_joint_vals);
    robot_state->update();
    moveit::core::robotStateToRobotStateMsg(*robot_state, req.start_state);

    // Goal constraint
    robot_state->setJointGroupPositions(joint_model_group, goal_joint_vals);
    robot_state->update();
    moveit_msgs::msg::Constraints goal_constraint = 
        kinematic_constraints::constructGoalConstraints(*robot_state, joint_model_group);
    req.goal_constraints.clear();
    req.goal_constraints.push_back(goal_constraint);

    // Set joint tolerance
    std::vector<moveit_msgs::msg::JointConstraint> joint_constraints = req.goal_constraints[0].joint_constraints;
    for (auto& joint_constraint : joint_constraints)
    {
        joint_constraint.tolerance_above = 0.001;
        joint_constraint.tolerance_below = 0.001;
    }

    planning_interface::PlannerManagerPtr planner_manager = planning_pipeline->getPlannerManager();

    // list of available planners
    std::vector<std::string> algorithms;
    planner_manager->getPlanningAlgorithms(algorithms);
    for (const auto& algorithm : algorithms)
    {
        RCLCPP_INFO(lerp_example_node->get_logger(), "Available planner: %s", algorithm.c_str());
    }

    // =================== planning ===================

    // Use planning pipeline
    // planning_pipeline->displayComputedMotionPlans(false);
    planning_pipeline->generatePlan(psm->getPlanningScene(), req, res);

    // // Use planning context
    // planning_interface::PlanningContextPtr context = planner_manager->getPlanningContext(psm->getPlanningScene(), req, res.error_code_);
    // if (context)
    // {
    //     context->solve(res);
    // }

    // deallocate the robot_state
    robot_state.reset();

    // stop the node
    rclcpp::shutdown();

    return 0;
}
#include <rclcpp/rclcpp.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include "moveit/planning_interface/planning_interface.h"
#include "moveit/robot_state/conversions.h"
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
// #include <moveit_visual_tools/moveit_visual_tools.h>
// #include <moveit/robot_state/joint_model_group.hpp>

#include "multiply.h"
#include <cuda_runtime.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("foliation_example");

int main(int argc, char** argv)
{
    const std::string GROUP_NAME = "arm";

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto foliation_example_node = rclcpp::Node::make_shared("foliation_example_node", node_options);

    // print out the node name
    RCLCPP_INFO(LOGGER, "Node name: %s", foliation_example_node->get_name());

    // Create a robot model
    robot_model_loader::RobotModelLoaderPtr robot_model_loader(
      new robot_model_loader::RobotModelLoader(foliation_example_node, "robot_description"));
    const moveit::core::RobotModelPtr& kinematic_model = robot_model_loader->getModel();
    if (kinematic_model == nullptr)
    {
        RCLCPP_ERROR(foliation_example_node->get_logger(), "Failed to load robot model");
        return 1;
    }

    // Using the RobotModelLoader, we can construct a planning scene monitor that
    // will create a planning scene, monitors planning scene diffs, and apply the diffs to it's
    // internal planning scene. We then call startSceneMonitor, startWorldGeometryMonitor and
    // startStateMonitor to fully initialize the planning scene monitor
    planning_scene_monitor::PlanningSceneMonitorPtr psm(
        new planning_scene_monitor::PlanningSceneMonitor(foliation_example_node, robot_model_loader));

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

    // Set the robot state to the current state
    robot_state->setToDefaultValues();
    
    // Get the joint model group
    const moveit::core::JointModelGroup* joint_model_group = robot_state->getJointModelGroup(GROUP_NAME);

    // Get the joint names
    const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();

    // Print the joint names
    for (const auto& joint_name : joint_names)
    {
        RCLCPP_INFO(foliation_example_node->get_logger(), "Joint name: %s", joint_name.c_str());
    }

    // Print End Effector Link
    RCLCPP_INFO(foliation_example_node->get_logger(), "End Effector Link: %s", joint_model_group->getLinkModelNames().back().c_str());

    // Get link names from the joint model group
    const std::vector<std::string>& link_names_from_joint_model_group = joint_model_group->getLinkModelNames();
    for (const auto& link_name : link_names_from_joint_model_group)
    {
        RCLCPP_INFO(foliation_example_node->get_logger(), "Link name from joint model group: %s", link_name.c_str());
    }

    // test forward kinematics
    robot_state->setToRandomPositions(joint_model_group);
    robot_state->update();

    // print joint values
    std::vector<double> sampled_joint_values;
    robot_state->copyJointGroupPositions(joint_model_group, sampled_joint_values);
    for (const auto& joint_value : sampled_joint_values)
    {
        RCLCPP_INFO(foliation_example_node->get_logger(), "Joint value: %f", joint_value);
    }

    std::cout << "link poses:" << std::endl;
    // print poses of each link of link_names_from_joint_model_group
    for (const auto& link_name : link_names_from_joint_model_group)
    {
        const Eigen::Isometry3d& link_pose = robot_state->getGlobalLinkTransform(link_name);
        std::cout << link_name << std::endl;
        std::cout << "trans:" << std::endl;
        std::cout << link_pose.translation().transpose() << std::endl;
        std::cout << "rotation:" << std::endl;
        std::cout << link_pose.rotation() << std::endl;
    }

    // try to use cuda to compute the forward kinematics
    

    // stop the node
    rclcpp::shutdown();

    return 0;
}
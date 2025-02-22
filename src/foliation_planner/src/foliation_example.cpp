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

static const rclcpp::Logger LOGGER = rclcpp::get_logger("foliation_example");

int main(int argc, char** argv)
{
    const std::string GROUP_NAME = "arm";
    const std::string PLANNER_ID = "RRG";

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

    /*********************************************** Add Obstacles ***********************************************/

    // Add a collision object to the planning scene
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = kinematic_model->getModelFrame();
    collision_object.id = "box1";

    // Define a box to add to the world
    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = 0.5;
    primitive.dimensions[1] = 0.5;
    primitive.dimensions[2] = 0.5;

    // Define a pose for the box (specified relative to frame_id)
    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = 0.7;
    box_pose.position.y = 0.0;
    box_pose.position.z = 0.8;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;

    // Add the collision object to the planning scene
    moveit_msgs::msg::PlanningScene planning_scene_msg;
    planning_scene_msg.world.collision_objects.push_back(collision_object);
    planning_scene_msg.is_diff = true;
    psm->newPlanningSceneMessage(planning_scene_msg);

    /*****************************************************************************************************/
    
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

    // Create pipeline, where third argumentn is parameter namespace
    planning_pipeline::PlanningPipelinePtr planning_pipeline(
      new planning_pipeline::PlanningPipeline(kinematic_model, foliation_example_node, "foliation"));

    planning_interface::MotionPlanRequest req;
    planning_interface::MotionPlanResponse res;

    // Set the start and goal joint values
    std::vector<double> start_joint_vals = {1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> goal_joint_vals = {-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    req.group_name = GROUP_NAME;

    req.planner_id = PLANNER_ID;

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
        RCLCPP_INFO(foliation_example_node->get_logger(), "Available planner: %s", algorithm.c_str());
    }

    // // =================== planning ===================

    // Use planning pipeline
    // planning_pipeline->displayComputedMotionPlans(false);
    planning_pipeline->generatePlan(psm->getPlanningScene(), req, res);

    /* Check that the planning was successful */
    if (res.error_code_.val == res.error_code_.SUCCESS)
    {
        RCLCPP_INFO(foliation_example_node->get_logger(), "Compute plan successfully");
    }
    else
    {
        RCLCPP_INFO(foliation_example_node->get_logger(), "Could not compute plan successfully");
    }


    // // Use planning context
    // planning_interface::PlanningContextPtr context = planner_manager->getPlanningContext(psm->getPlanningScene(), req, res.error_code_);
    // if (context)
    // {
    //     context->solve(res);
    // }

    std::shared_ptr<rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>> display_publisher =
      foliation_example_node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);
    std::shared_ptr<rclcpp::Publisher<moveit_msgs::msg::PlanningScene>> planning_scene_publisher =
      foliation_example_node->create_publisher<moveit_msgs::msg::PlanningScene>("/planning_scene", 1);

    moveit_msgs::msg::DisplayTrajectory display_trajectory;
    /* Visualize the trajectory */
    moveit_msgs::msg::MotionPlanResponse response;
    res.getMessage(response);

    display_trajectory.trajectory_start = response.trajectory_start;
    display_trajectory.trajectory.push_back(response.trajectory);
    
    // use loop to publish the trajectory
    while (rclcpp::ok())
    {
        display_publisher->publish(display_trajectory);

        // publish the planning scene
        planning_scene_publisher->publish(planning_scene_msg);
        
        rclcpp::spin_some(foliation_example_node);

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));

    }

    // deallocate the robot_state
    robot_state.reset();

    // stop the node
    rclcpp::shutdown();

    return 0;
}
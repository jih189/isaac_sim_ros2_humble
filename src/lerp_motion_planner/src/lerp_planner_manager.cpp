#include "lerp_motion_planner/lerp_planner_manager.hpp"

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"
#include "pluginlib/class_list_macros.hpp"

#include "lerp_motion_planner/lerp_planning_context.hpp"

namespace lerp_interface
{

bool LerpPlannerManager::initialize(
  const moveit::core::RobotModelConstPtr & model,
  const rclcpp::Node::SharedPtr & node,
  const std::string & parameter_namespace)
{
  node_ = node;
  for (const std::string & group_name : model->getJointModelGroupNames()) {
    planning_contexts_[group_name] =
      std::make_shared<LerpPlanningContext>("lerp_planning_context", group_name, node);
  }
  static_cast<void>(model);  // Suppress "unused" warning.
  static_cast<void>(parameter_namespace);  // Suppress "unused" warning.
  return true;
}

std::string LerpPlannerManager::getDescription() const
{
  return "LERP for Moveit2";
}

void LerpPlannerManager::getPlanningAlgorithms(std::vector<std::string> & algs) const
{
  algs.clear();
  algs.push_back("lerp");
}

planning_interface::PlanningContextPtr LerpPlannerManager::getPlanningContext(
  const planning_scene::PlanningSceneConstPtr & planning_scene,
  const planning_interface::MotionPlanRequest & req,
  moveit_msgs::msg::MoveItErrorCodes & error_code) const
{
  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

  if (!planning_scene) {
    RCLCPP_ERROR(node_->get_logger(), "No planning scene supplied as input");
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
    return planning_interface::PlanningContextPtr();
  }

  if (req.group_name.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No group specified to plan for");
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
    return planning_interface::PlanningContextPtr();
  }

  // Retrieve and configure existing context.
  const std::shared_ptr<LerpPlanningContext> & context = planning_contexts_.at(req.group_name);

  context->setPlanningScene(planning_scene);
  context->setMotionPlanRequest(req);

  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

  return context;
}

void LerpPlannerManager::setPlannerConfigurations(
  const planning_interface::PlannerConfigurationMap & pcs)
{
  planner_configs_ = pcs;
}

}  // namespace lerp_interface

// Register the `LerpPlannerManager` class as a plugin.
PLUGINLIB_EXPORT_CLASS(
  lerp_interface::LerpPlannerManager,
  planning_interface::PlannerManager)
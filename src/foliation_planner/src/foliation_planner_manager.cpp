#include "foliation_planner/foliation_planner_manager.hpp"

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"
#include "pluginlib/class_list_macros.hpp"

#include "foliation_planner/foliation_planning_context.hpp"

namespace foliation_interface
{

bool FoliationPlannerManager::initialize(
  const moveit::core::RobotModelConstPtr & model,
  const rclcpp::Node::SharedPtr & node,
  const std::string & parameter_namespace)
{
  node_ = node;
  for (const std::string & group_name : model->getJointModelGroupNames()) {
    planning_contexts_[group_name] =
      std::make_shared<FoliationPlanningContext>("foliation_planning_context", group_name, node, model, parameter_namespace);
  }
  return true;
}

std::string FoliationPlannerManager::getDescription() const
{
  return "Foliation for Moveit2";
}

void FoliationPlannerManager::getPlanningAlgorithms(std::vector<std::string> & algs) const
{
  algs.clear();
  algs.push_back("foliation");
}

planning_interface::PlanningContextPtr FoliationPlannerManager::getPlanningContext(
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
  const std::shared_ptr<FoliationPlanningContext> & context = planning_contexts_.at(req.group_name);

  context->setPlanningScene(planning_scene);
  context->setMotionPlanRequest(req);

  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

  return context;
}

void FoliationPlannerManager::setPlannerConfigurations(
  const planning_interface::PlannerConfigurationMap & pcs)
{
  planner_configs_ = pcs;
}

}  // namespace foliation_interface

// Register the `FoliationPlannerManager` class as a plugin.
PLUGINLIB_EXPORT_CLASS(
  foliation_interface::FoliationPlannerManager,
  planning_interface::PlannerManager)
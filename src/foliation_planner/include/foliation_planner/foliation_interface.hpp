#ifndef FOLIATION_INTERFACE_H
#define FOLIATION_INTERFACE_H

#include <memory>

#include "moveit/planning_interface/planning_interface.h"
#include "rclcpp/rclcpp.hpp"

namespace foliation_interface
{

class FoliationInterface
{
public:
  FoliationInterface(const rclcpp::Node::SharedPtr & node, const std::string & group_name, const moveit::core::RobotModelConstPtr & model,
                const std::string & parameter_namespace)
  : node_(node),
    group_name_(group_name),
    robot_model_(model),
    parameter_namespace_(parameter_namespace)
  {
    loadPlannerConfigurations();
    num_steps_ = 10;
  }

  bool solve(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & request,
    planning_interface::MotionPlanDetailedResponse & response);

  bool planner_busy = false;

protected:
  // /** @brief Load planner configurations for specified group into planner_config */
  // bool loadPlannerConfiguration(const std::string & group_name, const std::string & planner_id,
  //                               const std::map<std::string, std::string> &  group_params,
  //                               planning_interface::PlannerConfigurationSettings & planner_config);

  /** @brief Configure the planners*/
  void loadPlannerConfigurations();

private:
  std::shared_ptr<rclcpp::Node> node_;
  std::string group_name_;
  moveit::core::RobotModelConstPtr robot_model_;
  std::string parameter_namespace_;

  // Planner parameters
  int num_steps_;

  int dof_;

  void interpolate(moveit::core::RobotStatePtr& robot_state,
                   const moveit::core::JointModelGroup* joint_model_group, const std::vector<double>& start_joint_vals,
                   const std::vector<double>& goal_joint_vals, robot_trajectory::RobotTrajectoryPtr& joint_trajectory);
};

}  // namespace foliation_interface

#endif  // FOLIATION_INTERFACE_H
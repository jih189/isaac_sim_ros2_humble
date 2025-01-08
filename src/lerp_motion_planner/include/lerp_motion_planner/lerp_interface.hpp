#ifndef LERP_INTERFACE_H
#define LERP_INTERFACE_H

#include <memory>

#include "moveit/planning_interface/planning_interface.h"
#include "rclcpp/rclcpp.hpp"

namespace lerp_interface
{

class LerpInterface
{
public:
  LerpInterface(const rclcpp::Node::SharedPtr & node)
  : node_(node)
  {
    num_steps_ = 10;
  }

  bool solve(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & request,
    planning_interface::MotionPlanDetailedResponse & response);

  bool planner_busy = false;

private:
  std::shared_ptr<rclcpp::Node> node_;

  int num_steps_;
  int dof_;

  void interpolate(moveit::core::RobotStatePtr& robot_state,
                   const moveit::core::JointModelGroup* joint_model_group, const std::vector<double>& start_joint_vals,
                   const std::vector<double>& goal_joint_vals, robot_trajectory::RobotTrajectoryPtr& joint_trajectory);
};

}  // namespace lerp_interface

#endif  // LERP_INTERFACE_H
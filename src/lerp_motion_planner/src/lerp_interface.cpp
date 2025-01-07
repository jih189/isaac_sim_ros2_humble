#include "lerp_motion_planner/lerp_interface.hpp"

#include <chrono>
#include <memory>

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"
#include "moveit/robot_state/conversions.h"
#include "rclcpp/rclcpp.hpp"

namespace lerp_interface
{

namespace
{

constexpr unsigned NumSteps = 10;

}  // namespace

bool LerpInterface::solve(
  const planning_scene::PlanningSceneConstPtr & planning_scene,
  const planning_interface::MotionPlanRequest & request,
  planning_interface::MotionPlanDetailedResponse & response)
{
  RCLCPP_INFO(node_->get_logger(), "Planning trajectory");

  auto result_traj = std::make_shared<robot_trajectory::RobotTrajectory>(
    planning_scene->getRobotModel(), request.group_name);

  rclcpp::Time start_time = node_->now();

  response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
  response.processing_time_.clear();
  response.processing_time_.push_back(node_->now().seconds() - start_time.seconds());
  response.description_.clear();
  response.trajectory_.clear();

  response.trajectory_.push_back(result_traj);

  return true;
}

}  // namespace lerp_interface
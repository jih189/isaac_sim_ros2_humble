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
  }

  bool solve(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & request,
    planning_interface::MotionPlanDetailedResponse & response);

  bool planner_busy = false;

private:
  std::shared_ptr<rclcpp::Node> node_;
};

}  // namespace lerp_interface

#endif  // LERP_INTERFACE_H
#ifndef LERP_PLANNING_CONTEXT_H
#define LERP_PLANNING_CONTEXT_H

#include <memory>
#include <string>

#include "moveit/planning_interface/planning_interface.h"

#include "lerp_motion_planner/lerp_interface.hpp"

namespace lerp_interface
{

class LerpPlanningContext : public planning_interface::PlanningContext
{
public:
  LerpPlanningContext(
    const std::string & context_name,
    const std::string & group_name,
    const rclcpp::Node::SharedPtr & node)
  : planning_interface::PlanningContext(context_name, group_name),
    lerp_interface_(std::make_shared<LerpInterface>(node))
  {
  }

  ~LerpPlanningContext() override
  {
  }

  bool solve(planning_interface::MotionPlanResponse & res) override;

  bool solve(planning_interface::MotionPlanDetailedResponse & res) override;

  bool terminate() override
  {
    return true;
  }

  void clear() override
  {
  }

private:
  std::shared_ptr<LerpInterface> lerp_interface_;
};

}  // lerp_interface

#endif  // LERP_PLANNING_CONTEXT_H
#ifndef FOLIATION_PLANNING_CONTEXT_H
#define FOLIATION_PLANNING_CONTEXT_H

#include <memory>
#include <string>

#include "moveit/planning_interface/planning_interface.h"

#include "foliation_planner/foliation_interface.hpp"

namespace foliation_interface
{

class FoliationPlanningContext : public planning_interface::PlanningContext
{
public:
  FoliationPlanningContext(
    const std::string & context_name,
    const std::string & group_name,
    const rclcpp::Node::SharedPtr & node,
    const moveit::core::RobotModelConstPtr & model,
    const std::string & parameter_namespace)
  : planning_interface::PlanningContext(context_name, group_name),
    foliation_interface_(std::make_shared<FoliationInterface>(node, group_name, model, parameter_namespace))
  {
  }

  ~FoliationPlanningContext() override
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
  std::shared_ptr<FoliationInterface> foliation_interface_;
};

}  // foliation_interface

#endif  // FOLIATION_PLANNING_CONTEXT_H
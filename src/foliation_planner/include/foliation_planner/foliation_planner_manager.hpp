#ifndef FOLIATION_PLANNER_MANAGER_H
#define FOLIATION_PLANNER_MANAGER_H

#include <map>
#include <string>
#include <vector>

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"

#include "foliation_planner/foliation_planning_context.hpp"

namespace foliation_interface
{

class FoliationPlannerManager : public planning_interface::PlannerManager
{

public:
  FoliationPlannerManager()
  {
  }

  bool initialize(
    const moveit::core::RobotModelConstPtr & model,
    const rclcpp::Node::SharedPtr & node,
    const std::string & parameter_namespace) override;

  bool canServiceRequest(const planning_interface::MotionPlanRequest & req) const override
  {
    return req.trajectory_constraints.constraints.empty();
  }

  std::string getDescription() const override;

  void getPlanningAlgorithms(std::vector<std::string> & algs) const override;

  planning_interface::PlanningContextPtr getPlanningContext(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & req,
    moveit_msgs::msg::MoveItErrorCodes & error_code) const override;

  void setPlannerConfigurations(const planning_interface::PlannerConfigurationMap & pcs) override;

private:
  std::shared_ptr<rclcpp::Node> node_;
  std::map<std::string, std::shared_ptr<FoliationPlanningContext>> planning_contexts_;
  planning_interface::PlannerConfigurationMap planner_configs_;
};

}  // namespace foliation_interface

#endif  // FOLIATION_PLANNER_MANAGER_H
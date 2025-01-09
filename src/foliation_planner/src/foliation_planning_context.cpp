#include "foliation_planner/foliation_planning_context.hpp"

namespace foliation_interface
{

bool FoliationPlanningContext::solve(planning_interface::MotionPlanDetailedResponse & res)
{
  return foliation_interface_->solve(planning_scene_, request_, res);
}

bool FoliationPlanningContext::solve(planning_interface::MotionPlanResponse & res)
{
  planning_interface::MotionPlanDetailedResponse res_detailed;
  bool planning_success = solve(res_detailed);

  res.error_code_ = res_detailed.error_code_;

  if (planning_success) {
    res.trajectory_ = res_detailed.trajectory_[0];
    res.planning_time_ = res_detailed.processing_time_[0];
  }

  return planning_success;
}

}  // namespace foliation_interface
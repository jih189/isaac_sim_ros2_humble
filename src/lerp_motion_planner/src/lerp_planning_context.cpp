#include "lerp_motion_planner/lerp_planning_context.hpp"

namespace lerp_interface
{

bool LerpPlanningContext::solve(planning_interface::MotionPlanDetailedResponse & res)
{
  return lerp_interface_->solve(planning_scene_, request_, res);
}

bool LerpPlanningContext::solve(planning_interface::MotionPlanResponse & res)
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

}  // namespace lerp_interface
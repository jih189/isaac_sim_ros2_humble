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

  // std::cout << "==========================================================" << std::endl;
  // std::cout << "Robot model name: " << planning_scene->getRobotModel()->getName() << std::endl;
  // std::cout << "Group name: " << request.group_name << std::endl;
  // std::cout << "number of joints of group: " << planning_scene->getRobotModel()->getJointModelGroup(request.group_name)->getVariableCount() << std::endl;
  // std::cout << "==========================================================" << std::endl;

  const moveit::core::JointModelGroup* joint_model_group = planning_scene->getRobotModel()->getJointModelGroup(request.group_name);
  dof_ = joint_model_group->getVariableCount();

  moveit::core::RobotModelConstPtr robot_model = planning_scene->getRobotModel();
  moveit::core::RobotStatePtr start_state(new moveit::core::RobotState(robot_model));

  // Extract start state from request
  start_state->setToDefaultValues();
  moveit::core::robotStateMsgToRobotState(request.start_state, *start_state);
  start_state->update();

  // Extract state values from request
  std::vector<double> start_joint_vals;
  start_state->copyJointGroupPositions(request.group_name, start_joint_vals);

  // Extract goal state from request
  std::vector<double> goal_joint_vals;
  for(const auto& joint_constraint : request.goal_constraints[0].joint_constraints)
  {
    goal_joint_vals.push_back(joint_constraint.position);
  }
  
  // Create a robot trajectory
  robot_trajectory::RobotTrajectoryPtr result_traj = std::make_shared<robot_trajectory::RobotTrajectory>(
    planning_scene->getRobotModel(), request.group_name);

  rclcpp::Time start_time = node_->now();

  // Interpolate between start and goal states
  interpolate(start_state, joint_model_group, start_joint_vals, goal_joint_vals, result_traj);
  
  response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  response.processing_time_.clear();
  response.processing_time_.push_back(node_->now().seconds() - start_time.seconds());
  response.description_.clear();
  response.description_.push_back("LERP motion planner");
  response.trajectory_.clear();
  response.trajectory_.push_back(result_traj);

  return true;
}

void LerpInterface::interpolate(moveit::core::RobotStatePtr& rob_state,
                                const moveit::core::JointModelGroup* joint_model_group,
                                const std::vector<double>& start_joint_vals, const std::vector<double>& goal_joint_vals,
                                robot_trajectory::RobotTrajectoryPtr& joint_trajectory)
{
  joint_trajectory->clear();

  std::vector<double> dt_vector;
  for (int joint_index = 0; joint_index < dof_; ++joint_index)
  {
    double dt = (goal_joint_vals[joint_index] - start_joint_vals[joint_index]) / num_steps_;
    dt_vector.push_back(dt);
  }

  for (int step = 0; step <= num_steps_; ++step)
  {
    std::vector<double> joint_values;
    for (int k = 0; k < dof_; ++k)
    {
      double joint_value = start_joint_vals[k] + step * dt_vector[k];
      joint_values.push_back(joint_value);
    }
    rob_state->setJointGroupPositions(joint_model_group, joint_values);
    rob_state->update();

    // Add the state to the trajectory
    joint_trajectory->addSuffixWayPoint(rob_state, 0.1);
  }
}

}  // namespace lerp_interface
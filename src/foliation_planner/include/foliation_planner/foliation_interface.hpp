#ifndef FOLIATION_INTERFACE_H
#define FOLIATION_INTERFACE_H


#include <chrono>
#include <memory>

#include "foliation_planner/robot_info.hpp"

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"
#include "moveit/robot_state/conversions.h"
#include "rclcpp/rclcpp.hpp"

// cudampl include
#include <CUDAMPLib/spaces/SingleArmSpace.h>
#include <CUDAMPLib/constraints/EnvConstraintSphere.h>
#include <CUDAMPLib/constraints/EnvConstraintCuboid.h>
#include <CUDAMPLib/constraints/EnvConstraintCylinder.h>
#include <CUDAMPLib/constraints/SelfCollisionConstraint.h>
#include <CUDAMPLib/constraints/TaskSpaceConstraint.h>
#include <CUDAMPLib/constraints/BoundaryConstraint.h>
#include <CUDAMPLib/tasks/SingleArmTask.h>
#include <CUDAMPLib/planners/RRG.h>
#include <CUDAMPLib/termination/StepTermination.h>
#include <CUDAMPLib/termination/TimeoutTermination.h>

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

    // check if the parameter exists
    if (!node->has_parameter("collision_spheres_file_path"))
    {
      RCLCPP_ERROR(node->get_logger(), "collision_spheres_file_path not found");
    }
    else
    {
      // load the collision spheres file path
      std::string collision_spheres_file_path;
      node->get_parameter("collision_spheres_file_path", collision_spheres_file_path);

      // Create robot info
      robot_info_ptr_ = std::make_shared<RobotInfo>(model, group_name, collision_spheres_file_path);

      // Create self collision constraint
      self_collision_constraint_ = std::make_shared<CUDAMPLib::SelfCollisionConstraint>(
          "self_collision_constraint",
          robot_info_ptr_->getCollisionSpheresMap(),
          robot_info_ptr_->getCollisionSpheresRadius(),
          robot_info_ptr_->getSelfCollisionEnabledMap()
      );
    }
  }

  bool solve(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & request,
    planning_interface::MotionPlanDetailedResponse & response);

protected:
  /** @brief Load planner configurations for specified group into planner_config */
  bool loadPlannerConfiguration(const std::string & group_name, const std::string & planner_id,
                                planning_interface::PlannerConfigurationSettings & planner_config);

  /** @brief Configure the planners*/
  void loadPlannerConfigurations();

  planning_interface::PlannerConfigurationMap planner_configs_;

private:
  std::shared_ptr<rclcpp::Node> node_;
  std::string group_name_;
  moveit::core::RobotModelConstPtr robot_model_;
  std::string parameter_namespace_;

  // Planner parameters
  float obstacle_sphere_radius_;

  // dof
  int dof_;

  // cudampl objects
  CUDAMPLib::SelfCollisionConstraintPtr self_collision_constraint_;
  std::shared_ptr<RobotInfo> robot_info_ptr_;

  bool solve_motion_task(
    moveit::core::RobotStatePtr& robot_state,
    const moveit::core::JointModelGroup* joint_model_group, 
    const std::vector<shapes::ShapeConstPtr> obstacle_shapes,
    const std::vector<Eigen::Affine3d> obstacle_poses,
    const std::vector<double>& start_joint_vals, std::vector<CUDAMPLib::BaseConstraintPtr> goal_constraints,
    robot_trajectory::RobotTrajectoryPtr& joint_trajectory,
    float max_planning_time);
};

}  // namespace foliation_interface

#endif  // FOLIATION_INTERFACE_H
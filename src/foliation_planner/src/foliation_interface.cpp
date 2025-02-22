#include "foliation_planner/foliation_interface.hpp"

#include <chrono>
#include <memory>

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"
#include "moveit/robot_state/conversions.h"
#include "rclcpp/rclcpp.hpp"

// #include <fstream> // for debugging

namespace foliation_interface
{

namespace
{

constexpr unsigned NumSteps = 10;

}  // namespace

bool FoliationInterface::solve(
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

  const collision_detection::WorldConstPtr world = planning_scene->getWorld();

  // Generate point cloud for all obstacle shapes
  std::vector<Eigen::Vector3d> obstacle_points;
  for (std::string object_name : world->getObjectIds())
  {
    auto object_in_world = world->getObject(object_name);
    for(size_t i = 0; i < object_in_world->shapes_.size(); ++i)
    {
      auto pc = shapeToPointCloud(object_in_world->shapes_[i], object_in_world->pose_ * object_in_world->shape_poses_[i], 0.05);
      obstacle_points.insert(obstacle_points.end(), pc.begin(), pc.end());
    }
  }

  // // create an obj file for debugging
  // std::string obj_file_name = "/home/ros/ros2_ws/src/obstacles.obj";
  // std::ofstream obj_file(obj_file_name);
  // for (const auto& point : obstacle_points)
  // {
  //   obj_file << "v " << point[0] << " " << point[1] << " " << point[2] << std::endl;
  // }
  // obj_file.close();

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
  response.description_.push_back("Foliation planner");
  response.trajectory_.clear();
  response.trajectory_.push_back(result_traj);

  return true;
}

void FoliationInterface::interpolate(moveit::core::RobotStatePtr& rob_state,
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
    joint_trajectory->addSuffixWayPoint(std::make_shared<moveit::core::RobotState>(*rob_state), 0.1);
  }
}

std::vector<Eigen::Vector3d> FoliationInterface::shapeToPointCloud(const shapes::ShapeConstPtr& shape, const Eigen::Isometry3d& pose, float resolution)
{
  std::vector<Eigen::Vector3d> point_cloud;

  // Generate points evenly distributed on the surface of the shape using resolution
  if (shape->type == shapes::BOX)
  {
    const shapes::Box* box = dynamic_cast<const shapes::Box*>(shape.get());
    Eigen::Vector3d half_size(box->size[0] / 2, box->size[1] / 2, box->size[2] / 2);
    
    // X-faces: x = ±half_size.x
    for (double y = -half_size[1]; y <= half_size[1]; y += resolution)
    {
      for (double z = -half_size[2]; z <= half_size[2]; z += resolution)
      {
        Eigen::Vector3d point1 = pose * Eigen::Vector3d(half_size[0], y, z);
        Eigen::Vector3d point2 = pose * Eigen::Vector3d(-half_size[0], y, z);
        point_cloud.push_back(point1);
        point_cloud.push_back(point2);
      }
    }

    // Y-faces: y = ±half_size.y
    for (double x = -half_size[0]; x <= half_size[0]; x += resolution)
    {
      for (double z = -half_size[2]; z <= half_size[2]; z += resolution)
      {
        Eigen::Vector3d point1 = pose * Eigen::Vector3d(x, half_size[1], z);
        Eigen::Vector3d point2 = pose * Eigen::Vector3d(x, -half_size[1], z);
        point_cloud.push_back(point1);
        point_cloud.push_back(point2);
      }
    }

    // Z-faces: z = ±half_size.z
    for (double x = -half_size[0]; x <= half_size[0]; x += resolution)
    {
      for (double y = -half_size[1]; y <= half_size[1]; y += resolution)
      {
        Eigen::Vector3d point1 = pose * Eigen::Vector3d(x, y, half_size[2]);
        Eigen::Vector3d point2 = pose * Eigen::Vector3d(x, y, -half_size[2]);
        point_cloud.push_back(point1);
        point_cloud.push_back(point2);
      }
    }
  }
  else if (shape->type == shapes::SPHERE)
  {
    const shapes::Sphere* sphere = dynamic_cast<const shapes::Sphere*>(shape.get());
    Eigen::Vector3d center = pose.translation();
    double radius = sphere->radius;
    for (double theta = 0; theta <= M_PI; theta += resolution)
    {
      for (double phi = 0; phi <= 2 * M_PI; phi += resolution)
      {
        Eigen::Vector3d point = center + Eigen::Vector3d(
            radius * sin(theta) * cos(phi),
            radius * sin(theta) * sin(phi),
            radius * cos(theta));
        point_cloud.push_back(point);
      }
    }
  }
  else if (shape->type == shapes::CYLINDER)
  {
    const shapes::Cylinder* cylinder = dynamic_cast<const shapes::Cylinder*>(shape.get());
    Eigen::Vector3d center = pose.translation();
    double radius = cylinder->radius;
    double half_length = cylinder->length / 2;
    for (double theta = 0; theta <= 2 * M_PI; theta += resolution)
    {
      for (double z = -half_length; z <= half_length; z += resolution)
      {
        Eigen::Vector3d point = center + Eigen::Vector3d(radius * cos(theta), radius * sin(theta), z);
        point_cloud.push_back(point);
      }
    }

    // Top face: z = half_length
    for (double r = 0; r <= radius; r += resolution)
    {
      for (double theta = 0; theta <= 2 * M_PI; theta += resolution)
      {
        // Only add points on the boundary (i.e., when r is near the radius)
        // Adjust the condition if you strictly want the outer edge.
        if (fabs(r - radius) < resolution || r == 0)  // sample boundary and center
        {
          Eigen::Vector3d point = center + Eigen::Vector3d(r * cos(theta), r * sin(theta), half_length);
          point_cloud.push_back(point);
        }
      }
    }

    // Bottom face: z = -half_length
    for (double r = 0; r <= radius; r += resolution)
    {
      for (double theta = 0; theta <= 2 * M_PI; theta += resolution)
      {
        if (fabs(r - radius) < resolution || r == 0)
        {
          Eigen::Vector3d point = center + Eigen::Vector3d(r * cos(theta), r * sin(theta), -half_length);
          point_cloud.push_back(point);
        }
      }
    }
  }
  else
  {
    RCLCPP_ERROR(node_->get_logger(), "Shape type not supported");
  }

  return point_cloud;
}

void FoliationInterface::loadPlannerConfigurations()
{
  if (node_->has_parameter(parameter_namespace_ + ".planner_configs.foliation.num_step")){
    RCLCPP_INFO(node_->get_logger(), "foliation planner_configs.num_step found");
    const rclcpp::Parameter parameter = node_->get_parameter(parameter_namespace_ + ".planner_configs.foliation.num_step");
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER){
      num_steps_ = parameter.as_int();
    }
    else{
      RCLCPP_ERROR(node_->get_logger(), "foliation planner_configs.num_step is not an integer");
    }
  }
  else{
    RCLCPP_ERROR(node_->get_logger(), "foliation planner_configs.num_step not found");
  }
  
  // // print the parameter value
  // RCLCPP_INFO(node_->get_logger(), "Parameter value: %s", parameter.value_to_string().c_str());
}

}  // namespace foliation_interface
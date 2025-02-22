#include "foliation_planner/foliation_interface.hpp"

// #include <fstream> // for debugging

namespace foliation_interface
{

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

  // Generate point cloud from world
  std::vector<Eigen::Vector3d> obstacle_points = genPointCloudFromWorld(planning_scene->getWorld());

  // Extract start state from request
  moveit::core::RobotStatePtr start_state(new moveit::core::RobotState(planning_scene->getRobotModel()));
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

  // Solve the problem
  // We need to check if the environment collision obstacles is changed or not. If so, then
  // we need to reinitialize the planner.
  bool is_solved = solve_motion_task(start_state, joint_model_group, obstacle_points, start_joint_vals, goal_joint_vals, result_traj, request.allowed_planning_time);

  std::cout << "Planning done" << std::endl;

  if (is_solved)
  {
    RCLCPP_INFO(node_->get_logger(), "Task is solved");
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    response.processing_time_.clear();
    response.processing_time_.push_back(node_->now().seconds() - start_time.seconds());
    response.description_.clear();
    response.description_.push_back("Foliation planner");
    response.trajectory_.clear();
    response.trajectory_.push_back(result_traj);
  }
  else
  {
    RCLCPP_ERROR(node_->get_logger(), "Task is failed to solve");
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
    response.processing_time_.clear();
    response.processing_time_.push_back(node_->now().seconds() - start_time.seconds());
    response.description_.clear();
    response.description_.push_back("Foliation planner");
    response.trajectory_.clear();
  }

  return true;
}

bool FoliationInterface::solve_motion_task(
    moveit::core::RobotStatePtr& robot_state,
    const moveit::core::JointModelGroup* joint_model_group, 
    const std::vector<Eigen::Vector3d> obstacle_points,
    const std::vector<double>& start_joint_vals, 
    const std::vector<double>& goal_joint_vals, 
    robot_trajectory::RobotTrajectoryPtr& joint_trajectory,
    float max_planning_time
    )
{
  joint_trajectory->clear();

  // Prepare collision spheres
  std::vector<std::vector<float>> obstacle_points_float;
  for (const auto& point : obstacle_points)
  {
    obstacle_points_float.push_back({(float)point[0], (float)point[1], (float)point[2]});
  }

  std::vector<float> obstacle_sphere_radius;
  for (size_t i = 0; i < obstacle_points.size(); ++i)
  {
    obstacle_sphere_radius.push_back(obstacle_sphere_radius_ * 2);
  }

  // construct environment constraint
  CUDAMPLib::EnvConstraintPtr env_constraint = std::make_shared<CUDAMPLib::EnvConstraint>(
    "obstacle_constraint",
    obstacle_points_float,
    obstacle_sphere_radius
  );

  // create constraints
  std::vector<CUDAMPLib::BaseConstraintPtr> constraints;
  constraints.push_back(self_collision_constraint_);
  constraints.push_back(env_constraint);

  // create space
  CUDAMPLib::SingleArmSpacePtr single_arm_space = std::make_shared<CUDAMPLib::SingleArmSpace>(
    robot_info_ptr_->getDimension(),
    constraints,
    robot_info_ptr_->getJointTypes(),
    robot_info_ptr_->getJointPoses(),
    robot_info_ptr_->getJointAxes(),
    robot_info_ptr_->getLinkMaps(),
    robot_info_ptr_->getCollisionSpheresMap(),
    robot_info_ptr_->getCollisionSpheresPos(),
    robot_info_ptr_->getCollisionSpheresRadius(),
    robot_info_ptr_->getActiveJointMap(),
    robot_info_ptr_->getLowerBounds(),
    robot_info_ptr_->getUpperBounds(),
    robot_info_ptr_->getDefaultJointValues(),
    robot_info_ptr_->getLinkNames(),
    0.02 // resolution
  );

  // Convert start and goal joint values to float
  std::vector<std::vector<float>> start_joint_values_set;
  std::vector<float> start_joint_vals_float;
  for (const auto& val : start_joint_vals)
  {
    start_joint_vals_float.push_back(val);
  }
  start_joint_values_set.push_back(start_joint_vals_float);

  std::vector<std::vector<float>> goal_joint_values_set;
  std::vector<float> goal_joint_vals_float;
  for (const auto& val : goal_joint_vals)
  {
    goal_joint_vals_float.push_back(val);
  }
  goal_joint_values_set.push_back(goal_joint_vals_float);

  // create the task
  CUDAMPLib::SingleArmTaskPtr problem_task = std::make_shared<CUDAMPLib::SingleArmTask>(
      start_joint_values_set,
      goal_joint_values_set
  );

  // create the planner
  CUDAMPLib::RRGPtr planner = std::make_shared<CUDAMPLib::RRG>(single_arm_space);

  // set the task
  planner->setMotionTask(problem_task, true);

  // solve the task
  CUDAMPLib::TimeoutTerminationPtr termination_condition = std::make_shared<CUDAMPLib::TimeoutTermination>(max_planning_time);
  planner->solve(termination_condition);

  bool found_solution = problem_task->hasSolution();

  if (problem_task->hasSolution())
  {
    std::vector<std::vector<float>> solution_path = problem_task->getSolution();

    // generate robot trajectory msg
    for (size_t i = 0; i < solution_path.size(); i++)
    {
      // convert solution_path[i] to double vector
      std::vector<double> solution_path_i_double = std::vector<double>(solution_path[i].begin(), solution_path[i].end());
      robot_state->setJointGroupPositions(joint_model_group, solution_path_i_double);
      robot_state->update();

      joint_trajectory->addSuffixWayPoint(std::make_shared<moveit::core::RobotState>(*robot_state), 0.1);
    }
  }

  // reset the environment constraint, planner, and space
  env_constraint.reset();
  planner.reset();
  single_arm_space.reset();
  problem_task.reset();

  return found_solution;
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

std::vector<Eigen::Vector3d> FoliationInterface::genPointCloudFromWorld(const collision_detection::WorldConstPtr & world)
{
  // Generate point cloud for all obstacle shapes
  std::vector<Eigen::Vector3d> obstacle_points;
  for (std::string object_name : world->getObjectIds())
  {
    auto object_in_world = world->getObject(object_name);
    for(size_t i = 0; i < object_in_world->shapes_.size(); ++i)
    {
      auto pc = shapeToPointCloud(object_in_world->shapes_[i], object_in_world->pose_ * object_in_world->shape_poses_[i], 2 * obstacle_sphere_radius_);
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

  return obstacle_points;
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
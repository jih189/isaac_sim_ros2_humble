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

  if (request.goal_constraints.size() != 1)
  {
    // print error in red
    std::cout << "\033[1;31m" << "For now, only one goal constraint is supported" << "\033[0m" << std::endl;
    return false;
  }
  
  //////////////////////////////////////// Convert moveit goal constraints to CUDAMPLib goal constraints ////////////////////////////////////////
  std::vector<CUDAMPLib::BaseConstraintPtr> goal_constraints;

  // Create boundary constraints for goal constraints.
  std::vector<std::string> joint_names = robot_info_ptr_->getJointNames();
  std::vector<bool> active_joint_map = robot_info_ptr_->getActiveJointMap();
  std::vector<float> goal_constraint_lower_bounds;
  std::vector<float> goal_constraint_upper_bounds;
  std::vector<float> default_lower_bounds = robot_info_ptr_->getLowerBounds();
  std::vector<float> default_upper_bounds = robot_info_ptr_->getUpperBounds();

  for (size_t i = 0; i < joint_names.size(); i++)
  {
    // check if this joint is in the goal constraints
    bool is_in_goal_constraints = false;
    for (const auto& joint_constraint : request.goal_constraints[0].joint_constraints)
    {
      if (joint_constraint.joint_name == joint_names[i])
      {
        // if current joint is in the goal constraints, set bounds from goal constraints
        is_in_goal_constraints = true;
        // set bounds from goal constraints
        goal_constraint_lower_bounds.push_back(joint_constraint.position - joint_constraint.tolerance_below);
        goal_constraint_upper_bounds.push_back(joint_constraint.position + joint_constraint.tolerance_above);
        break;
      }
    }

    if (!is_in_goal_constraints)
    {
      // set bounds from default bounds
      goal_constraint_lower_bounds.push_back(default_lower_bounds[i]);
      goal_constraint_upper_bounds.push_back(default_upper_bounds[i]);
    }
  }

  CUDAMPLib::BoundaryConstraintPtr goal_boundary_constraint = std::make_shared<CUDAMPLib::BoundaryConstraint>(
      "goal_boundary_constraint",
      goal_constraint_lower_bounds,
      goal_constraint_upper_bounds,
      robot_info_ptr_->getActiveJointMap()
  );

  goal_constraints.push_back(goal_boundary_constraint);

  if (request.goal_constraints[0].position_constraints.size() > 0 || request.goal_constraints[0].orientation_constraints.size() > 0)
  {
    if (request.goal_constraints[0].position_constraints.size() > 1 || request.goal_constraints[0].orientation_constraints.size() > 1)
    {
      // print error in red
      std::cout << "\033[1;31m" << "For now, only one position constraint and one orientation constraint are supported" << "\033[0m" << std::endl;
      return false;
    }

    if (
      request.goal_constraints[0].position_constraints.size() == 1 && 
      request.goal_constraints[0].orientation_constraints.size() == 1 &&
      request.goal_constraints[0].position_constraints[0].link_name != request.goal_constraints[0].orientation_constraints[0].link_name
    )
    {
      // print error in red
      std::cout << "\033[1;31m" << "The link name of position constraint and orientation constraint should be the same" << "\033[0m" << std::endl;
      return false;
    }

    std::string task_link_name = "";

    if (request.goal_constraints[0].position_constraints.size() == 1)
    {
      task_link_name = request.goal_constraints[0].position_constraints[0].link_name;
    }
    
    if (request.goal_constraints[0].orientation_constraints.size() == 1)
    {
      task_link_name = request.goal_constraints[0].orientation_constraints[0].link_name;
    }

    if (task_link_name == "")
    {
      // print error in red
      std::cout << "\033[1;31m" << "link name of constraints is empty" << "\033[0m" << std::endl;
      return false;
    }

    int task_link_index = -1;
    std::vector<std::string> robot_link_names = robot_info_ptr_->getLinkNames();
    for (size_t i = 0; i < robot_link_names.size(); i++)
    {
        if (robot_link_names[i] == task_link_name)
        {
            task_link_index = i;
            break;
        }
    }

    if (task_link_index == -1)
    {
        // print error in red
        std::cout << "\033[1;31m" << "Link name is not found in the joint names" << "\033[0m" << std::endl;
        // print task link name
        std::cout << "Task link name: " << task_link_name << std::endl;

        // print link names
        std::cout << "Link names: ";
        for (size_t i = 0; i < robot_link_names.size(); i++)
        {
          std::cout << robot_link_names[i] << " ";
        }
        std::cout << std::endl;

        return false;
    }

    // set position and orientation constraints for default value.
    std::vector<float> reference_frame = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> tolerance = {1000, 1000, 1000, 10, 10, 10};

    if (request.goal_constraints[0].position_constraints.size() == 1)
    {
      float pos_x = request.goal_constraints[0].position_constraints[0].constraint_region.primitive_poses[0].position.x;
      float pos_y = request.goal_constraints[0].position_constraints[0].constraint_region.primitive_poses[0].position.y;
      float pos_z = request.goal_constraints[0].position_constraints[0].constraint_region.primitive_poses[0].position.z;

      reference_frame[0] = pos_x;
      reference_frame[1] = pos_y;
      reference_frame[2] = pos_z;

      float pos_x_tol = request.goal_constraints[0].position_constraints[0].constraint_region.primitives[0].dimensions[0];
      float pos_y_tol = request.goal_constraints[0].position_constraints[0].constraint_region.primitives[0].dimensions[1];
      float pos_z_tol = request.goal_constraints[0].position_constraints[0].constraint_region.primitives[0].dimensions[2];

      tolerance[0] = pos_x_tol;
      tolerance[1] = pos_y_tol;
      tolerance[2] = pos_z_tol;
    }

    if (request.goal_constraints[0].orientation_constraints.size() == 1)
    {
      // convert quaternion to rpy with Eigen
      Eigen::Quaterniond q(
        request.goal_constraints[0].orientation_constraints[0].orientation.w,
        request.goal_constraints[0].orientation_constraints[0].orientation.x,
        request.goal_constraints[0].orientation_constraints[0].orientation.y,
        request.goal_constraints[0].orientation_constraints[0].orientation.z
      );

      Eigen::Matrix3d rotation_matrix = q.toRotationMatrix();
      Eigen::Vector3d rpy = rotation_matrix.eulerAngles(0, 1, 2);

      reference_frame[3] = rpy[0];
      reference_frame[4] = rpy[1];
      reference_frame[5] = rpy[2];

      float roll_tol = request.goal_constraints[0].orientation_constraints[0].absolute_x_axis_tolerance;
      float pitch_tol = request.goal_constraints[0].orientation_constraints[0].absolute_y_axis_tolerance;
      float yaw_tol = request.goal_constraints[0].orientation_constraints[0].absolute_z_axis_tolerance;

      tolerance[3] = roll_tol;
      tolerance[4] = pitch_tol;
      tolerance[5] = yaw_tol;
    }

    // create offset matrix
    Eigen::Matrix4d offset_matrix = Eigen::Matrix4d::Identity();
    
    offset_matrix(0, 3) = request.goal_constraints[0].position_constraints[0].target_point_offset.x;
    offset_matrix(1, 3) = request.goal_constraints[0].position_constraints[0].target_point_offset.y;
    offset_matrix(2, 3) = request.goal_constraints[0].position_constraints[0].target_point_offset.z;

    // Convert moveit end effector pose constraints to CUDAMPLib task space constraints.
    CUDAMPLib::TaskSpaceConstraintPtr goal_task_space_constraint = std::make_shared<CUDAMPLib::TaskSpaceConstraint>(
        "goal_task_space_constraint",
        task_link_index,
        offset_matrix,
        reference_frame,
        tolerance
    );

    goal_constraints.push_back(goal_task_space_constraint);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Create a robot trajectory
  robot_trajectory::RobotTrajectoryPtr result_traj = std::make_shared<robot_trajectory::RobotTrajectory>(
    planning_scene->getRobotModel(), request.group_name);

  rclcpp::Time start_time = node_->now();

  // Solve the problem
  // We need to check if the environment collision obstacles is changed or not. If so, then
  // we need to reinitialize the planner.
  bool is_solved = solve_motion_task(start_state, joint_model_group, obstacle_points, start_joint_vals, goal_constraints, result_traj, request.allowed_planning_time);

  rclcpp::Time end_time = node_->now();

  if (is_solved)
  {
    RCLCPP_INFO(node_->get_logger(), "Task is solved");
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    response.processing_time_.clear();
    response.processing_time_.push_back(end_time.seconds() - start_time.seconds());
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
    response.processing_time_.push_back(end_time.seconds() - start_time.seconds());
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
    std::vector<CUDAMPLib::BaseConstraintPtr> goal_constraints,
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

  // std::vector<std::vector<float>> goal_joint_values_set;
  // std::vector<float> goal_joint_vals_float;
  // for (const auto& val : goal_joint_vals)
  // {
  //   goal_joint_vals_float.push_back(val);
  // }
  // goal_joint_values_set.push_back(goal_joint_vals_float);

  // create space for goal region
  goal_constraints.push_back(env_constraint);
  goal_constraints.push_back(self_collision_constraint_);

  CUDAMPLib::SingleArmSpacePtr goal_region = std::make_shared<CUDAMPLib::SingleArmSpace>(
    robot_info_ptr_->getDimension(),
    goal_constraints,
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

  // create the task
  CUDAMPLib::SingleArmTaskPtr problem_task = std::make_shared<CUDAMPLib::SingleArmTask>(
      start_joint_values_set,
      goal_region
  );

  // create the planner
  CUDAMPLib::RRGPtr planner = std::make_shared<CUDAMPLib::RRG>(single_arm_space);

  // find the planner configuration from planner_configs_
  std::string planner_config_name = joint_model_group->getName() + "[RRGConfigDefault]";
  if (planner_configs_.find(planner_config_name) != planner_configs_.end())
  {
    RCLCPP_INFO(node_->get_logger(), "Found planner configuration '%s' and use them", planner_config_name.c_str());
    auto config_settings = planner_configs_[planner_config_name];

    if (config_settings.config.find("k") != config_settings.config.end())
    {
      planner->setK(std::stoi(config_settings.config["k"]));
    }

    if (config_settings.config.find("sample_attempts_in_each_iteration") != config_settings.config.end())
    {
      planner->setSampleAttemptsInEachIteration(std::stoi(config_settings.config["sample_attempts_in_each_iteration"]));
    }

    if (config_settings.config.find("max_travel_distance") != config_settings.config.end())
    {
      planner->setMaxTravelDistance(std::stof(config_settings.config["max_travel_distance"]));
    }
  }

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
  // Load obstacle configurations
  if (node_->has_parameter(parameter_namespace_ + ".obstacle_configs.obstacle_sphere_radius")){
    const rclcpp::Parameter parameter = node_->get_parameter(parameter_namespace_ + ".obstacle_configs.obstacle_sphere_radius");
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE){
      obstacle_sphere_radius_ = parameter.as_double();
      if (obstacle_sphere_radius_ <= 0){
        RCLCPP_ERROR(node_->get_logger(), "OBSTACLE CONFIGURATION obstacle_configs.obstacle_sphere_radius must be greater than 0");
      }
      else{
        RCLCPP_INFO(node_->get_logger(), "OBSTACLE CONFIGURATION obstacle_configs.obstacle_sphere_radius: %f", obstacle_sphere_radius_);
      }
    }
    else{
      RCLCPP_ERROR(node_->get_logger(), "OBSTACLE CONFIGURATION obstacle_configs.obstacle_sphere_radius is not an double");
    }
  }
  else{
    RCLCPP_ERROR(node_->get_logger(), "OBSTACLE CONFIGURATION obstacle_configs.obstacle_sphere_radius not found");
  }
  
  // reset planner_configs_ which is a map of group name to planner configuration settings
  planner_configs_.clear();

  // read the planning configuration for each group
  for (const std::string& group_name : robot_model_->getJointModelGroupNames())
  {
    const std::string group_name_param = parameter_namespace_ + "." + group_name;

    // get parameters specific to each planner type
    std::vector<std::string> config_names;

    if (node_->get_parameter(group_name_param + ".planner_configs", config_names))
    {
      for (const auto& planner_id : config_names)
      {
        planning_interface::PlannerConfigurationSettings pc;
        if (loadPlannerConfiguration(group_name, planner_id, pc))
        {
          planner_configs_[pc.name] = pc;
        }
      }
    }
  }
}

bool FoliationInterface::loadPlannerConfiguration(const std::string& group_name, const std::string& planner_id,
                                             planning_interface::PlannerConfigurationSettings& planner_config)
{
  rcl_interfaces::msg::ListParametersResult planner_params_result =
      node_->list_parameters({ parameter_namespace_ + ".planner_configs." + planner_id }, 2);

  if (planner_params_result.names.empty())
  {
    RCLCPP_ERROR(node_->get_logger(), "Could not find the planner configuration '%s' on the param server", planner_id.c_str());
    return false;
  }

  planner_config.name = group_name + "[" + planner_id + "]";
  planner_config.group = group_name;

  planner_config.config = std::map<std::string, std::string>();

  // read parameters specific for this configuration
  for (const auto& planner_param : planner_params_result.names)
  {
    const rclcpp::Parameter param = node_->get_parameter(planner_param);
    auto param_name = planner_param.substr(planner_param.find(planner_id) + planner_id.size() + 1);
    planner_config.config[param_name] = param.value_to_string();
  }

  return true;
}

}  // namespace foliation_interface
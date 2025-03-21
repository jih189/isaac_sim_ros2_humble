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

  // Get shapes from the world
  std::vector<shapes::ShapeConstPtr> obstacle_shapes;
  std::vector<Eigen::Affine3d> obstacle_poses;
  for (const auto& object_name : planning_scene->getWorld()->getObjectIds())
  {
    auto obj = planning_scene->getWorld()->getObject(object_name);

    for (size_t i = 0; i < obj->shapes_.size(); ++i)
    {
      obstacle_shapes.push_back(obj->shapes_[i]);
      obstacle_poses.push_back(obj->pose_ * obj->shape_poses_[i]);
    }
  }

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
  bool is_solved = solve_motion_task(start_state, joint_model_group, obstacle_shapes, obstacle_poses, start_joint_vals, goal_constraints, result_traj, request.allowed_planning_time);

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

    return true;
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

    return false;
  }
}

bool FoliationInterface::solve_motion_task(
    moveit::core::RobotStatePtr& robot_state,
    const moveit::core::JointModelGroup* joint_model_group, 
    const std::vector<shapes::ShapeConstPtr> obstacle_shapes,
    const std::vector<Eigen::Affine3d> obstacle_poses,
    const std::vector<double>& start_joint_vals, 
    std::vector<CUDAMPLib::BaseConstraintPtr> goal_constraints,
    robot_trajectory::RobotTrajectoryPtr& joint_trajectory,
    float max_planning_time
    )
{
  joint_trajectory->clear();

   std::vector<CUDAMPLib::BaseConstraintPtr> constraints;

  // Need to classify the obstacle shapes
  std::vector<shapes::ShapeConstPtr> obstacle_boxes;
  std::vector<Eigen::Affine3d> obstacle_poses_boxes;
  std::vector<shapes::ShapeConstPtr> obstacle_spheres;
  std::vector<Eigen::Affine3d> obstacle_poses_spheres;
  std::vector<shapes::ShapeConstPtr> obstacle_cylinders;
  std::vector<Eigen::Affine3d> obstacle_poses_cylinders;

  for (size_t i = 0; i < obstacle_shapes.size(); i++)
  {
    if (obstacle_shapes[i]->type == shapes::BOX)
    {
      obstacle_boxes.push_back(obstacle_shapes[i]);
      obstacle_poses_boxes.push_back(obstacle_poses[i]);
    }
    else if (obstacle_shapes[i]->type == shapes::SPHERE)
    {
      obstacle_spheres.push_back(obstacle_shapes[i]);
      obstacle_poses_spheres.push_back(obstacle_poses[i]);
    }
    else if (obstacle_shapes[i]->type == shapes::CYLINDER)
    {
      obstacle_cylinders.push_back(obstacle_shapes[i]);
      obstacle_poses_cylinders.push_back(obstacle_poses[i]);
    }
  }

  if (obstacle_boxes.size() > 0)
  {
    // create obstacle constraint for boxes
    std::vector<std::vector<float>> bounding_boxes_pos; // [(x, y, z), ...]
    std::vector<std::vector<float>> bounding_boxes_orientation_matrix; // [(r11, r12, r13, r21, r22, r23, r31, r32, r33), ...]
    std::vector<std::vector<float>> bounding_boxes_max; // [(x_max, y_max, z_max), ...]
    std::vector<std::vector<float>> bounding_boxes_min; // [(x_min, y_min, z_min), ...]

    // for each box
    for (size_t i = 0; i < obstacle_boxes.size(); i++)
    {
      // extract position
      Eigen::Vector3d pos = obstacle_poses_boxes[i].translation();
      bounding_boxes_pos.push_back({(float)pos[0], (float)pos[1], (float)pos[2]});

      // extract orientation
      Eigen::Matrix3d rotation_matrix = obstacle_poses_boxes[i].rotation();
      bounding_boxes_orientation_matrix.push_back({
        (float)rotation_matrix(0, 0), (float)rotation_matrix(0, 1), (float)rotation_matrix(0, 2),
        (float)rotation_matrix(1, 0), (float)rotation_matrix(1, 1), (float)rotation_matrix(1, 2),
        (float)rotation_matrix(2, 0), (float)rotation_matrix(2, 1), (float)rotation_matrix(2, 2)
      });

      // extract max and min
      const shapes::Box* box = dynamic_cast<const shapes::Box*>(obstacle_boxes[i].get());
      bounding_boxes_max.push_back({(float)( box->size[0] / 2), (float)( box->size[1] / 2), (float)( box->size[2] / 2)});
      bounding_boxes_min.push_back({(float)( -1 * box->size[0] / 2), (float)(-1 * box->size[1] / 2), (float)(-1 * box->size[2] / 2)});
    }

    CUDAMPLib::EnvConstraintCuboidPtr env_constraint_box = std::make_shared<CUDAMPLib::EnvConstraintCuboid>(
      "obstacle_box_constraint",
      bounding_boxes_pos,
      bounding_boxes_orientation_matrix,
      bounding_boxes_max,
      bounding_boxes_min
    );

    constraints.push_back(env_constraint_box);
  }

  if (obstacle_spheres.size() > 0)
  {
    // create obstacle constraint for spheres
    std::vector<std::vector<float>> obstacle_spheres_pos; // [(x, y, z), ...]
    std::vector<float> obstacle_spheres_radius; // [r1, r2, ...]

    // for each sphere
    for (size_t i = 0; i < obstacle_spheres.size(); i++)
    {
      // extract position
      Eigen::Vector3d pos = obstacle_poses_spheres[i].translation();
      obstacle_spheres_pos.push_back({(float)pos[0], (float)pos[1], (float)pos[2]});

      // extract radius
      const shapes::Sphere* sphere = dynamic_cast<const shapes::Sphere*>(obstacle_spheres[i].get());
      obstacle_spheres_radius.push_back((float)sphere->radius);
    }

    CUDAMPLib::EnvConstraintSpherePtr env_constraint_sphere = std::make_shared<CUDAMPLib::EnvConstraintSphere>(
      "obstacle_sphere_constraint",
      obstacle_spheres_pos,
      obstacle_spheres_radius
    );

    constraints.push_back(env_constraint_sphere);
  }

  if (obstacle_cylinders.size() > 0)
  {
    // create obstacle constraint for cylinders
    std::vector<std::vector<float>> obstacle_cylinders_pos; // [(x, y, z), ...]
    std::vector<std::vector<float>> obstacle_cylinders_orientation_matrix; // [(r11, r12, r13, r21, r22, r23, r31, r32, r33), ...]
    std::vector<float> obstacle_cylinders_radius; // [r1, r2, ...]
    std::vector<float> obstacle_cylinders_height; // [h1, h2, ...]

    // for each cylinder
    for (size_t i = 0; i < obstacle_cylinders.size(); i++)
    {
      // extract position
      Eigen::Vector3d pos = obstacle_poses_cylinders[i].translation();
      obstacle_cylinders_pos.push_back({(float)pos[0], (float)pos[1], (float)pos[2]});

      // extract orientation
      Eigen::Matrix3d rotation_matrix = obstacle_poses_cylinders[i].rotation();
      obstacle_cylinders_orientation_matrix.push_back({
        (float)rotation_matrix(0, 0), (float)rotation_matrix(0, 1), (float)rotation_matrix(0, 2),
        (float)rotation_matrix(1, 0), (float)rotation_matrix(1, 1), (float)rotation_matrix(1, 2),
        (float)rotation_matrix(2, 0), (float)rotation_matrix(2, 1), (float)rotation_matrix(2, 2)
      });

      // extract radius and height
      const shapes::Cylinder* cylinder = dynamic_cast<const shapes::Cylinder*>(obstacle_cylinders[i].get());
      obstacle_cylinders_radius.push_back((float)cylinder->radius);
      obstacle_cylinders_height.push_back((float)cylinder->length);
    }

    CUDAMPLib::EnvConstraintCylinderPtr env_constraint_cylinder = std::make_shared<CUDAMPLib::EnvConstraintCylinder>(
      "obstacle_cylinder_constraint",
      obstacle_cylinders_pos,
      obstacle_cylinders_orientation_matrix,
      obstacle_cylinders_radius,
      obstacle_cylinders_height
    );

    constraints.push_back(env_constraint_cylinder);
  }

  // add self collision constraint
  constraints.push_back(self_collision_constraint_);

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

  // create space for goal region
  // goal_constraints.push_back(env_constraint);
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
  else
  {
    // print error in red
    std::cout << "\033[1;31m" << "No solution found" << "\033[0m" << std::endl;
    std::cout << "Failure reason: " << problem_task->getFailureReason() << std::endl;
  }

  // // reset. Not sure is this necessary.
  // planner.reset();
  // single_arm_space.reset();
  // problem_task.reset();
  
  // for (size_t i = 0; i < constraints.size(); i++)
  // {
  //   constraints[i].reset();
  // }

  // for (size_t i = 0; i < goal_constraints.size(); i++)
  // {
  //   goal_constraints[i].reset();
  // }

  // goal_region.reset();

  return found_solution;
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
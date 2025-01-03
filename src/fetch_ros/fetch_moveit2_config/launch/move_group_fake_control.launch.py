import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
 
def generate_launch_description():

    # Constants for paths to different files and folders
    package_name_description = 'fetch_description'
    package_name_moveit_config = 'fetch_moveit2_config'

    # Set the path to different files and folders
    pkg_share_description = FindPackageShare(package=package_name_description).find(package_name_description)
    pkg_share_moveit_config = FindPackageShare(package=package_name_moveit_config).find(package_name_moveit_config)
 
    # Paths for various configuration files
    urdf_file_path = 'robots/fetch.urdf.xacro'
    srdf_file_path = 'config/fetch.srdf'
    moveit_controllers_file_path = 'config/controllers.yaml'
    joint_limits_file_path = 'config/joint_limits.yaml'
    kinematics_file_path = 'config/kinematics.yaml'
    pilz_cartesian_limits_file_path = 'config/pilz_cartesian_limits.yaml'
    ros2_controller_file_path = 'config/ros2_controllers.yaml'
 
    # Set the full paths
    urdf_model_path = os.path.join(pkg_share_description, urdf_file_path)
    srdf_model_path = os.path.join(pkg_share_moveit_config, srdf_file_path)
    moveit_controllers_file_path = os.path.join(pkg_share_moveit_config, moveit_controllers_file_path)
    joint_limits_file_path = os.path.join(pkg_share_moveit_config, joint_limits_file_path)
    kinematics_file_path = os.path.join(pkg_share_moveit_config, kinematics_file_path)
    pilz_cartesian_limits_file_path = os.path.join(pkg_share_moveit_config, pilz_cartesian_limits_file_path)
    ros2_controller_file_path = os.path.join(pkg_share_moveit_config, ros2_controller_file_path)
 
    # Load the robot configuration
    # Typically, you would also have this line in here: .robot_description(file_path=urdf_model_path)
    # Another launch file is launching the robot description.
    moveit_config = (
        MoveItConfigsBuilder("fetch", package_name=package_name_moveit_config)
        .robot_description(file_path=urdf_model_path, mappings={"use_fake_controller": "true"})
        .trajectory_execution(file_path=moveit_controllers_file_path)
        .robot_description_semantic(file_path=srdf_model_path)
        .joint_limits(file_path=joint_limits_file_path)
        .robot_description_kinematics(file_path=kinematics_file_path)
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"],
            default_planning_pipeline="ompl"
        )
        .planning_scene_monitor(
            publish_robot_description=False,
            publish_robot_description_semantic=True,
            publish_planning_scene=True,
        )
        .pilz_cartesian_limits(file_path=pilz_cartesian_limits_file_path)
        .to_moveit_configs()
    )

    print("==============================================")
    print("kinematics_file_path: ", moveit_config.robot_description_kinematics)
     
    # Start the actual move_group node/action server
    start_move_group_node_cmd = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
        ],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            {"publish_frequency": 10.0},
        ],
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            ros2_controller_file_path,
        ],
        output="both",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "controller_manager",],
    )

    arm_with_torso_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_with_torso_controller", "-c", "controller_manager"],
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Add any actions
    ld.add_action(start_move_group_node_cmd)
    ld.add_action(robot_state_publisher)
    ld.add_action(ros2_control_node)
    ld.add_action(joint_state_broadcaster_spawner)
    ld.add_action(arm_with_torso_controller_spawner)

    return ld
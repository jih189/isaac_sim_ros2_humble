import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import xacro

def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:
        return None
 
def generate_launch_description():
 
    # Constants for paths to different files and folders
    package_name_description = 'fetch_description'
    package_name_moveit_config = 'fetch_moveit2_config'
    package_name_bringup = 'fetch_bringup'
 
    # Set the path to different files and folders
    pkg_share_description = FindPackageShare(package=package_name_description).find(package_name_description)
    pkg_share_moveit_config = FindPackageShare(package=package_name_moveit_config).find(package_name_moveit_config)
    # pkg_share_bringup = FindPackageShare(package=package_name_bringup).find(package_name_bringup)
 
    # Paths for various configuration files
    urdf_file_path = 'robots/fetch.urdf.xacro'
    srdf_file_path = 'config/fetch.srdf'
    moveit_controllers_file_path = 'config/controllers.yaml'
    joint_limits_file_path = 'config/joint_limits.yaml'
    kinematics_file_path = 'config/kinematics.yaml'
    pilz_cartesian_limits_file_path = 'config/pilz_cartesian_limits.yaml'
    initial_positions_file_path = 'config/initial_positions.yaml'
    # ros_controllers_file_path = 'config/default_controllers.yaml'
    # rviz_config_file_path = 'rviz/move_group.rviz'
 
    # Set the full paths
    urdf_model_path = os.path.join(pkg_share_description, urdf_file_path)
    srdf_model_path = os.path.join(pkg_share_moveit_config, srdf_file_path)
    moveit_controllers_file_path = os.path.join(pkg_share_moveit_config, moveit_controllers_file_path)
    joint_limits_file_path = os.path.join(pkg_share_moveit_config, joint_limits_file_path)
    kinematics_file_path = os.path.join(pkg_share_moveit_config, kinematics_file_path)
    pilz_cartesian_limits_file_path = os.path.join(pkg_share_moveit_config, pilz_cartesian_limits_file_path)
    initial_positions_file_path = os.path.join(pkg_share_moveit_config, initial_positions_file_path)
    # ros_controllers_file_path = os.path.join(pkg_share_bringup, ros_controllers_file_path)
    # rviz_config_file = os.path.join(pkg_share_moveit_config, rviz_config_file_path)

    controller_params = os.path.join(get_package_share_directory(package_name_bringup), 'config', 'default_controllers.yaml')
 
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    use_fake_controller = LaunchConfiguration("use_fake_controller")
 
    # Declare the launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation clock if true')
 
    declare_use_rviz_cmd = DeclareLaunchArgument(
        name='use_rviz',
        default_value='false',
        description='Whether to start RViz')
    
    declare_use_fake_controller_cmd = DeclareLaunchArgument(
        name='use_fake_controller',
        default_value='true',
        description='Whether to use fake controller')
 
    # Load the robot configuration
    # Typically, you would also have this line in here: .robot_description(file_path=urdf_model_path)
    # Another launch file is launching the robot description.
    moveit_config = (
        MoveItConfigsBuilder("fetch", package_name=package_name_moveit_config)
        .robot_description(file_path=urdf_model_path)
        .trajectory_execution(file_path=moveit_controllers_file_path)
        .robot_description_semantic(file_path=srdf_model_path)
        .joint_limits(file_path=joint_limits_file_path)
        .robot_description_kinematics(file_path=kinematics_file_path)
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"],
            default_planning_pipeline="pilz_industrial_motion_planner"
        )
        .planning_scene_monitor(
            publish_robot_description=False,
            publish_robot_description_semantic=True,
            publish_planning_scene=True,
        )
        .pilz_cartesian_limits(file_path=pilz_cartesian_limits_file_path)
        .to_moveit_configs()
    )
     
    # Start the actual move_group node/action server
    start_move_group_node_cmd = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {'use_sim_time': use_sim_time},
            {'start_state': {'content': initial_positions_file_path}},
        ],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        condition=IfCondition(use_fake_controller),
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            {'use_sim_time': use_sim_time},
        ],
    )

    ros2_control_node = Node(
        package="controller_manager",
        # condition=IfCondition(use_fake_controller),
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            controller_params,
            {'use_sim_time': use_sim_time},
        ],
        output="both",
    )

    # joint_state_broadcaster_spawner = Node(
    #     package="controller_manager",
    #     condition=IfCondition(use_fake_controller),
    #     executable="spawner",
    #     arguments=[
    #         "joint_state_broadcaster",
    #         "--controller-manager",
    #         "controller_manager",
    #     ],
    # )

    # arm_controller_spawner = Node(
    #     package="controller_manager",
    #     condition=IfCondition(use_fake_controller),
    #     executable="spawner",
    #     arguments=["arm_controller", "-c", "controller_manager"],
    # )

    # torso_controller_spawner = Node(
    #     package="controller_manager",
    #     condition=IfCondition(use_fake_controller),
    #     executable="spawner",
    #     arguments=["torso_controller", "-c", "controller_manager"],
    # )
 
    # # RViz
    # start_rviz_node_cmd = Node(
    #     condition=IfCondition(use_rviz),
    #     package="rviz2",
    #     executable="rviz2",
    #     arguments=["-d", rviz_config_file],
    #     output="screen",
    #     parameters=[
    #         moveit_config.robot_description,
    #         moveit_config.robot_description_semantic,
    #         moveit_config.planning_pipelines,
    #         moveit_config.robot_description_kinematics,
    #         moveit_config.joint_limits,
    #         {'use_sim_time': use_sim_time}
    #     ],
    # )
     
    # exit_event_handler = RegisterEventHandler(
    #     condition=IfCondition(use_rviz),
    #     event_handler=OnProcessExit(
    #         target_action=start_rviz_node_cmd,
    #         on_exit=EmitEvent(event=Shutdown(reason='rviz exited')),
    #     ),
    # )
     
    # Create the launch description and populate
    ld = LaunchDescription()
 
    # Declare the launch options
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_use_fake_controller_cmd)
 
    # Add any actions
    ld.add_action(robot_state_publisher)
    ld.add_action(ros2_control_node)
    ld.add_action(start_move_group_node_cmd)
    # ld.add_action(joint_state_broadcaster_spawner)
    # ld.add_action(arm_controller_spawner)
    # ld.add_action(torso_controller_spawner)
    # ld.add_action(start_rviz_node_cmd)
     
    # # Clean shutdown of RViz
    # ld.add_action(exit_event_handler)
 
    return ld
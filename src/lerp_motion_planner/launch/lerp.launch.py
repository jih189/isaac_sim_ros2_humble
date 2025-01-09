import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
# import yaml


def generate_launch_description():

    package_name_description = 'fetch_description'
    package_name_moveit_config = 'fetch_moveit2_config'
    package_name_lerp_motion_planner = 'lerp_motion_planner'

    # Set the path to different files and folders
    pkg_share_description = FindPackageShare(package=package_name_description).find(package_name_description)
    pkg_share_moveit_config = FindPackageShare(package=package_name_moveit_config).find(package_name_moveit_config)
    
    urdf_file_path = 'robots/fetch.urdf.xacro'
    srdf_file_path = 'config/fetch.srdf'

    urdf_model_path = os.path.join(pkg_share_description, urdf_file_path)
    srdf_model_path = os.path.join(pkg_share_moveit_config, srdf_file_path)

    moveit_config = (
        MoveItConfigsBuilder("fetch", package_name=package_name_lerp_motion_planner)
        .robot_description(file_path=urdf_model_path, mappings={"use_fake_controller": "true"})
        .robot_description_semantic(file_path=srdf_model_path)
        .to_moveit_configs()
    )

    node_list = []

    lerp_example_node = Node(
        package='lerp_motion_planner',
        executable='lerp_example',
        name='lerp_example_node',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
        ]
    )

    # Launch RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
        ],
    )

    node_list.append(lerp_example_node)
    node_list.append(rviz_node)

    return LaunchDescription(node_list)
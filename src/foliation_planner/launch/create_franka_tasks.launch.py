import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():

    package_name_description = 'franka_description'
    package_name_moveit_config = 'franka_fr3_moveit_config'
    package_name_foliation_planner = 'foliation_planner'

    # Set the path to different files and folders
    pkg_share_description = FindPackageShare(package=package_name_description).find(package_name_description)
    pkg_share_moveit_config = FindPackageShare(package=package_name_moveit_config).find(package_name_moveit_config)
    pkg_share_foliation_planner = FindPackageShare(package=package_name_foliation_planner).find(package_name_foliation_planner)
    
    urdf_file_path = 'robots/fr3/fr3.urdf.xacro'
    srdf_file_path = 'srdf/fr3_arm.srdf.xacro'
    rviz_file_path = 'rviz/franka.rviz'
    moveit_controllers_file_path = 'config/fr3_controllers.yaml'
    collision_spheres_file_path = 'robots/franka.yml'

    urdf_model_path = os.path.join(pkg_share_description, urdf_file_path)
    srdf_model_path = os.path.join(pkg_share_moveit_config, srdf_file_path)
    rviz_config_path = os.path.join(pkg_share_foliation_planner, rviz_file_path)
    moveit_controllers_file_path = os.path.join(pkg_share_moveit_config, moveit_controllers_file_path)
    collision_spheres_file_path = os.path.join(pkg_share_description, collision_spheres_file_path)

    moveit_config = (
        MoveItConfigsBuilder("franka", package_name=package_name_foliation_planner)
        .robot_description(file_path=urdf_model_path, mappings={"use_fake_controller": "true"})
        .robot_description_semantic(file_path=srdf_model_path)
        .trajectory_execution(file_path=moveit_controllers_file_path)
        .to_moveit_configs()
    )

    node_list = []

    create_franka_motion_tasks_node = Node(
        package='foliation_planner',
        executable='create_motion_tasks',
        name='create_franka_motion_tasks_node',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            {
                "collision_spheres_file_path": collision_spheres_file_path,
                "group_name": "fr3_arm",
            }
        ]
    )

    # # Launch RViz
    # rviz_node = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     output="log",
    #     arguments=["-d", rviz_config_path],
    #     parameters=[
    #         moveit_config.robot_description,
    #         moveit_config.robot_description_semantic,
    #     ],
    # )

    node_list.append(create_franka_motion_tasks_node)
    # node_list.append(rviz_node)

    return LaunchDescription(node_list)
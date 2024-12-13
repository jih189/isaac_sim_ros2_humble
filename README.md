# ROS2 Humble Workspace in docker

## Description
We built this workspace to develop our ros package to interface with Isaac Sim. The src directory is mount to __/home/ros/ros2_ws/src__ in the container. Thus, you can place your ROS package there.

## Usage

### Build and enter container
```
cd docker

# build the image
./build.sh

# create the container
./run.sh

# you can create a new terminal to enter the container
./enter_container.sh

# delete the container
./delete_container.sh
```

### Build package
Once entering the container, the workspace src is mount to the your host machine, so you need to build the workspace in the container by
```
cd /home/ros/ros2_ws
colcon build
source install/setup.bash # this must be run for each terminal.
```

### Launch robot bringup

After you launch the similation container, IsaacSim has both joint state publisher and subscriber to publish the current joint state and receive command. You can check it by 
```
ros2 topic list | grep joint
``` 
And it should return topics like
```
/fetch_robot/joint_commands
/fetch_robot/joint_states
```
You can consider them as the API for you to control the robot. However, ROS2 has provide a standard interface, called ros controller, so other packages can easily be intergrated into your system. Therefore, you need to launch bringup to create the ros controller.
```
ros2 launch fetch_bringup fetch_bringup_launch.xml
```

Then you can open rviz2 to check the state of the robot.
```
rviz2
```

Then you can use following code to control the base
```
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_nav
```

### SLAM
If you want to build the build, you can use the slam system from NAV2. First, you need to launch the nav2 bringup in a new terminal

```
ros2 launch fetch_navigation nav.xml
```
then in another terminal you can launch the slam node
```
ros2 launch fetch_navigation slam.xml
```
By using the keyboard control, you can move the robot in the room, and the slam will generate the map. You can save the map by
```
ros2 run nav2_map_server map_saver_cli -f ~/map
```

If the robot does not build the map correctly, you may need to play with the parameter of the NAV2 and slam_tool in __($fetch_navigation)/config__

### Navigation
After you have a map and place it in __($fetch_navigation)/maps__, you can try to launch the nav2 stack. Before that, you must launch the rviz first, so you can initialize the robot pose. If you do not open rviz first, then you can see the map on it.
```
cd ros2_ws
rviz2 -d src/fetch_nav.rviz
```

Then you can launch nav2 stack in another terminal
```
ros2 launch fetch_navigation nav.xml
```

In the rviz2, you should see a map while the robot is empty there. This is because you did not set the initial pose of the robot, so it does not know where the robot actually is initially. You can set it by clicking "2D Pose Estimate" on the top bar, then in the map you can click and drag a direction to indicate the robot pose. Once you done that, the robot will be visible on the map.

Then, you can click 2D Goal Pose to select the target pose in the map, then the robot will navigate to there. You may need to play with those parameters to make the robot move smoothly.

## <span style="color:red">Warning</span>
Due to the design of Nav2, it can only publish the topic /cmd_vel_nav for controlling the robot, so we have to remap the differential_controller//differential_base_controller/cmd_vel_unstamped to it. As a result, you can have only one robot in the scene.
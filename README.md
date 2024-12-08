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
# ROS2 Humble Workspace in docker

## Description
We built this workspace to develop our ros package to interface with Isaac Sim. The src directory is mount to __/home/ros/ros2_ws/src__ in the container. Thus, you can place your ROS package there.

## Usage
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
#!/bin/bash
docker build -f Dockerfile.ros2 -t isaac-sim-ros2:latest .
docker build -f Dockerfile.ompl -t isaac-sim-ros2-ompl:latest .
docker build -f Dockerfile.moveit2 -t isaac-sim-ros2-moveit2:latest .
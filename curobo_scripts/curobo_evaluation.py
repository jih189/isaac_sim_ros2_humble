# Third Party
import torch

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.base import TensorDeviceType

from curobo.geom.types import WorldConfig, Sphere

import os
import yaml

class MotionPlanningTask:
    def __init__(self):
        self.start_joint_values = []
        self.goal_joint_values = []
        self.obstacle_pos = []
        self.radius = []

def load_motion_planning_tasks(task_dir_path):
    tasks = []
    metadata_path = os.path.join(task_dir_path, "metadata.txt")

    # Attempt to open the metadata file

    with open(metadata_path, 'r') as metadata:
        lines = metadata.readlines()

    # Process each task file listed in the metadata
    for line in lines:
        line = line.strip()
        if not line:
            continue

        task_file_path = os.path.join(task_dir_path, line)

        with open(task_file_path, 'r') as task_file:
            task_node = yaml.safe_load(task_file)

        task = MotionPlanningTask()

        # Load start and goal joint values
        task.start_joint_values = task_node["Start joint values"]
        task.goal_joint_values = task_node["Goal joint values"]

        # Load obstacles (if any)
        obstacles = task_node.get("Obstacles", [])
        for obstacle in obstacles:
            obstacle_pos = obstacle["Position"]
            obstacle_radius = obstacle["Radius"]
            task.obstacle_pos.append(obstacle_pos)
            task.radius.append(obstacle_radius)

        tasks.append(task)

    return tasks


# Load the tasks from the task directory
task_dir_path = "/home/motion_planning_tasks"
tasks = load_motion_planning_tasks(task_dir_path)

tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

# Create obstacle
obstacle_sphere = Sphere(
   name="sphere_1",
   radius=0.2,
   pose=[0.506, 0.0, 0.7, 1, 0, 0, 0],
   color=[0, 1.0, 0, 1.0],
)
world_model = WorldConfig(
   sphere=[obstacle_sphere],
)

# convert obstacle to cuboid. This is necessary for collision checking. I think this is stupid.
cuboid_world = WorldConfig.create_obb_world(world_model)

motion_gen_config = MotionGenConfig.load_from_robot_config(
    "/home/ros/curobo_scripts/fetch.yml",
    cuboid_world,
    interpolation_dt=0.01,
)

motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()


for t in tasks:
    pass
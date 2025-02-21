# Third Party
import torch

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.base import TensorDeviceType

from curobo.geom.types import WorldConfig, Sphere

import trimesh

import os
import yaml

def create_robot_spheres_meshes(kin_state):
    obstacle_spheres = kin_state.robot_spheres.cpu().numpy()[0]
    meshes = []
    for center_x, center_y, center_z, radius in obstacle_spheres:
        # Create an icosphere (you can adjust subdivisions for smoothness)
        sphere_mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        # Translate the sphere to its center
        sphere_mesh.apply_translation([center_x, center_y, center_z])
        meshes.append(sphere_mesh)
    return meshes

def create_robot_spheres_path_meshes(kin_state, step=50):
    waypoints_size = kin_state.robot_spheres.shape[0]
    meshes = []
    for i in range(0, waypoints_size, step):
        obstacle_spheres = kin_state.robot_spheres.cpu().numpy()[i]
        for center_x, center_y, center_z, radius in obstacle_spheres:
            # Create an icosphere (you can adjust subdivisions for smoothness)
            sphere_mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            # Translate the sphere to its center
            sphere_mesh.apply_translation([center_x, center_y, center_z])
            meshes.append(sphere_mesh)
    return meshes

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

# We need to create dummpy obstacle to initialize the motion generator, so it can first create the collision cache.
# Create obstacle
obstacle_sphere = Sphere(
   name="dummy_obstacle",
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
    collision_cache={"obb": 20, "mesh": 20}
)

motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

num_of_success = 0
num_of_tasks = 0
total_time_of_success_case = 0

for t in tasks:
    # update obstacle
    motion_gen.clear_world_cache()

    obstacle_spheres = []

    # print obstacles in task t
    for i in range(len(t.obstacle_pos)):
        # Create obstacle
        obstacle_sphere = Sphere(
            name="obstacle_" + str(i),
            radius=t.radius[i],
            pose=[t.obstacle_pos[i][0], t.obstacle_pos[i][1], t.obstacle_pos[i][2], 1, 0, 0, 0],
            color=[0, 1.0, 0, 1.0],
        )
        obstacle_spheres.append(obstacle_sphere)

    world_model = WorldConfig(
        sphere=obstacle_spheres,
    )

    # convert obstacle to cuboid.
    cuboid_world = WorldConfig.create_obb_world(world_model)

    motion_gen.update_world(cuboid_world)

    # print start and goal joint values
    print("Start joint values: ", t.start_joint_values)
    print("Goal joint values: ", t.goal_joint_values)

    start_state = JointState.from_position(
        tensor_args.to_device([t.start_joint_values]),
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
    )

    goal_state = JointState.from_position(
        tensor_args.to_device([t.goal_joint_values]),
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
    )

    # check if start and goal are valid
    valid_query, status = motion_gen.check_start_state(start_state)
    if not valid_query:
        print("\033[91m" + "Invalid start state" + "\033[0m")
        continue

    valid_query, status = motion_gen.check_start_state(goal_state)
    if not valid_query:
        print("\033[91m" + "Invalid goal state" + "\033[0m")
        continue

    result = motion_gen.plan_single_js(start_state, goal_state, MotionGenPlanConfig(
            enable_graph=True, 
            enable_opt=False,
            use_nn_ik_seed=False,
            need_graph_success=False,
            max_attempts=1000,
            timeout=100.0,
            enable_finetune_trajopt=False,
            parallel_finetune=False,
            finetune_attempts=0
        )
    )

    if result.success.cpu()[0]:
        # print all time taken
        print("Trajectory Generated: ", result.success)
        print("Time taken for graph_time: ", result.graph_time)
        print("Time taken for finetune_time: ", result.finetune_time)
        print("Time taken for trajopt_time: ", result.trajopt_time)
        print("Number of attempts: ", result.attempts)

        num_of_success += 1
        total_time_of_success_case += result.graph_time
    else:
        print("\033[91m" + "Failed to generate trajectory" + "\033[0m")
        print("Time taken for planning: ", result.graph_time)

    num_of_tasks += 1

    # start_kin_state = motion_gen.rollout_fn.compute_kinematics(start_state)
    # start_meshes = create_robot_spheres_meshes(start_kin_state)
    # # Create a list of trimesh sphere meshes
    # goal_kin_state = motion_gen.rollout_fn.compute_kinematics(goal_state)
    # goal_meshes = create_robot_spheres_meshes(goal_kin_state)

    # cuboid_world.save_world_as_mesh("obstacles.obj")
    # trimesh.util.concatenate(start_meshes).export('start_robot_spheres.obj')
    # trimesh.util.concatenate(goal_meshes).export('goal_robot_spheres.obj')

    # break    

print("Success rate: ", num_of_success / num_of_tasks)
print("Average time taken: ", total_time_of_success_case / num_of_success)
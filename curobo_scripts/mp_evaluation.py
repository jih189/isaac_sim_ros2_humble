# Third Party
import torch

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.base import TensorDeviceType

from curobo.geom.types import WorldConfig, Sphere
import trimesh

# include time
import time

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

start_state = JointState.from_position(
    tensor_args.to_device([[1.0, 1.0, 0.0, -1.0, 0.0, 0.6, 0.0]]),
    joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
)

goal_state = JointState.from_position(
    tensor_args.to_device([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]),
    joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
)

start_time = time.time()
result = motion_gen.plan_single_js(start_state, goal_state, MotionGenPlanConfig(
        enable_graph=True, 
        enable_opt=False,
        use_nn_ik_seed=False,
        need_graph_success=False,
        max_attempts=1,
        timeout=10.0,
        enable_finetune_trajopt=False,
        parallel_finetune=False,
        finetune_attempts=0
    )
)
end_time = time.time()
print("Time taken for planning: ", end_time - start_time)

# print all time taken
print("Time taken for graph_time: ", result.graph_time)
print("Time taken for finetune_time: ", result.finetune_time)
print("Time taken for trajopt_time: ", result.trajopt_time)
print("Number of attempts: ", result.attempts)

# check the number of valid states in graph
traj = result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps
print("Trajectory Generated: ", result.success)

path_states = motion_gen.rollout_fn.compute_kinematics(traj)
path_meshes = create_robot_spheres_path_meshes(path_states, step=500)
# Create a list of trimesh sphere meshes
start_kin_state = motion_gen.rollout_fn.compute_kinematics(start_state)
start_meshes = create_robot_spheres_meshes(start_kin_state)
# Create a list of trimesh sphere meshes
goal_kin_state = motion_gen.rollout_fn.compute_kinematics(goal_state)
goal_meshes = create_robot_spheres_meshes(goal_kin_state)

cuboid_world.save_world_as_mesh("obstacles.obj")
trimesh.util.concatenate(start_meshes).export('start_robot_spheres.obj')
trimesh.util.concatenate(goal_meshes).export('goal_robot_spheres.obj')
trimesh.util.concatenate(path_meshes).export('path_robot_spheres.obj')
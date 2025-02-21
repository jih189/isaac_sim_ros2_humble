# Third Party
import torch
# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.types import WorldConfig, Cuboid, Mesh, Capsule, Cylinder, Sphere
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.geom.sdf.utils import create_collision_checker

import trimesh

# Create obstacle
obstacle_sphere = Sphere(
   name="sphere_1",
   radius=0.2,
   pose=[0.506, 0.0, 0.50606, 1, 0, 0, 0],
   color=[0, 1.0, 0, 1.0],
)
world_model = WorldConfig(
   sphere=[obstacle_sphere],
)

# convert obstacle to cuboid. This is necessary for collision checking. I think this is stupid.
cuboid_world = WorldConfig.create_obb_world(world_model)

tensor_args = TensorDeviceType()

# robot world config
robot_file = "/home/ros/curobo_scripts/fetch.yml"
robot_world_config = RobotWorldConfig.load_from_config(robot_file, cuboid_world, collision_activation_distance=0.0)
# create curobo_fn
curobo_fn = RobotWorld(robot_world_config)

# set test_joint_values
test_joint_values = [
    [0, 0, 0, 0, 0.0, 0.0, 0.0],
]
# create q_s based on test_joint_values
q_s = torch.tensor(test_joint_values, **(tensor_args.as_torch_dict()))

# check collision
d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q_s)
print("d_world", d_world)
print("d_self", d_self)

# compute kinematics
kin_state = curobo_fn.get_kinematics(q_s)
# get link spheres of first state in q_s
obstacle_spheres = kin_state.link_spheres_tensor.cpu().numpy()[0]

# Create a list of trimesh sphere meshes
meshes = []
for center_x, center_y, center_z, radius in obstacle_spheres:
    # Create an icosphere (you can adjust subdivisions for smoothness)
    sphere_mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    # Translate the sphere to its center
    sphere_mesh.apply_translation([center_x, center_y, center_z])
    meshes.append(sphere_mesh)

# Merge all sphere meshes into one scene
combined = trimesh.util.concatenate(meshes)

# Export to an OBJ file
cuboid_world.save_world_as_mesh("obstacles.obj")
combined.export('robot_spheres.obj')
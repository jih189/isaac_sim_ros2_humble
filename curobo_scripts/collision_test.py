# Third Party
import torch
# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.types import WorldConfig, Cuboid, Mesh, Capsule, Cylinder, Sphere

import matplotlib.pyplot as plt
import numpy as np

def plot_sphere(ax, center, radius, color='b', alpha=0.6):
    # Create a grid of angles
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    # Parametric equations for the sphere surface
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    # Plot the surface
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

robot_file = "/home/ros/curobo_scripts/fetch.yml"

obstacle_sphere = Sphere(
   name="sphere_1",
   radius=0.2,
   pose=[0.456, 0.758833, 0.50606, 1, 0, 0, 0],
   color=[0, 1.0, 0, 1.0],
)

world_model = WorldConfig(
   sphere=[obstacle_sphere],
)

tensor_args = TensorDeviceType()
robot_world_config = RobotWorldConfig.load_from_config(robot_file, world_model, collision_activation_distance=0.0)

print(robot_world_config.world_model._env_mesh_names)

# curobo_fn = RobotWorld(robot_world_config)

# test_joint_values = [
#     [1.0, 0.2, 0.4, 0.3, 0.0, 0.0, 0.0],
# ]

# # create q_s based on test_joint_values
# q_s = torch.tensor(test_joint_values, **(tensor_args.as_torch_dict()))

# b, dof = q_s.shape
# kin_state = curobo_fn.get_kinematics(q_s)
# spheres = kin_state.link_spheres_tensor.view(b, 1, -1, 4)
# d_world = curobo_fn.get_collision_constraint(spheres, 0)
# print("d_world", d_world)


# print("robot_model_state", robot_model_state.ee_position, robot_model_state.ee_quaternion)

# print(robot_model_state.link_spheres_tensor.cpu().numpy()[0])

# obstacle_spheres = robot_model_state.link_spheres_tensor.cpu().numpy()[0]

# # d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q_s)

# # print("d_world", d_world)
# # print("d_self", d_self)

# # Set up the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot each sphere
# for sphere in obstacle_spheres:
#     center = sphere[:3]
#     radius = sphere[3]
#     plot_sphere(ax, center, radius)

# # Set labels and show plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
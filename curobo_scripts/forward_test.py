# Third Party
import torch

# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

import numpy as np

# convenience function to store tensor type and device
tensor_args = TensorDeviceType()

# this example loads urdf from a configuration file, you can also load from path directly
# load a urdf, the base frame and the end-effector frame:
config_file = load_yaml("/home/ros/curobo_scripts/fetch.yml")

urdf_file = config_file["robot_cfg"]["kinematics"][
    "urdf_path"
]  # Send global path starting with "/"
base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

# Generate robot configuration from  urdf path, base frame, end effector frame

robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

kin_model = CudaRobotModel(robot_cfg.kinematics)

joint_names = robot_cfg.kinematics.cspace.joint_names

print("joint names", joint_names)

active_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']

active_joint_values = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    [0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
]

# get indices of active joints
active_joint_indices = []
for j in active_joint_names:
    active_joint_indices.append(joint_names.index(j))

# create q based on active joint values and names, for un-mentioned joints, set to 0.0. active_joint_values is a list of list
q = torch.zeros((3, len(joint_names)), **(tensor_args.as_torch_dict()))

for i in range(len(active_joint_values)):
    for j in range(len(active_joint_names)):
        q[i, active_joint_indices[j]] = active_joint_values[i][j]

out = kin_model.get_state(q)

# convert tensor to numpy
out_position = out.ee_pose.position.cpu().numpy()
out_quaternion = out.ee_pose.quaternion.cpu().numpy()

for i in range(out_position.shape[0]):
    print(f"Pose {i}:")
    print(f"Position: {out_position[i]}")
    print(f"Quaternion: {out_quaternion[i]}")
    print()
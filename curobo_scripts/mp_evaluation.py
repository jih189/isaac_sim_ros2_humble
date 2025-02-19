# Third Party
import torch

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.base import TensorDeviceType

tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

world_config = {
    "cuboid": {
        "table": {
            "dims": [0.1, 0.1, 0.1],  # x, y, z
            "pose": [12.0, 0.0, 0.0, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
        },
    },
}

motion_gen_config = MotionGenConfig.load_from_robot_config(
    "/home/ros/curobo_scripts/fetch.yml",
    world_config,
    interpolation_dt=0.01,
)

motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

start_state = JointState.from_position(
    tensor_args.to_device([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
)

goal_state = JointState.from_position(
    tensor_args.to_device([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]),
    joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
)

result = motion_gen.plan_single_js(start_state, goal_state, MotionGenPlanConfig(enable_graph=True, need_graph_success=True))

# check the number of valid states in graph
traj = result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps
print("Trajectory Generated: ", result.success)
      
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from sim2real_assets import FOURIOR_CFG_WITH_HAND_FIX, FOURIOR_WITH_HAND_FIX_URDF_PATH, FOURIOR_CFG_WITH_HAND_FIX_PAYLOAD, FOURIOR_WITH_HAND_FIX_PAYLOAD_URDF_PATH # type: ignore

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


from .motions.joint_names import ROBOT_JOINT_NAME_DICT_URDF_FOURIOR

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

ROBOT_DICT = {
    "fourior_with_hand_fix": {"model": FOURIOR_CFG_WITH_HAND_FIX, "motion_dir": "fourior", "urdf_path": FOURIOR_WITH_HAND_FIX_URDF_PATH},
    "fourior_with_hand_fix_payload": {"model": FOURIOR_CFG_WITH_HAND_FIX_PAYLOAD, "motion_dir": "fourior", "urdf_path": FOURIOR_WITH_HAND_FIX_PAYLOAD_URDF_PATH},
}

@configclass
class HumanoidOperatorEnvCfg(DirectRLEnvCfg):
    robot_name: str = "fourior_with_hand_fix_payload"
    compute_eq_torque = False

    if 'urdf_path' in ROBOT_DICT[robot_name]:
        urdf_model_path = ROBOT_DICT[robot_name]["urdf_path"]
        package_dirs = os.path.dirname(urdf_model_path)
        urdf_joint_name = ROBOT_JOINT_NAME_DICT_URDF_FOURIOR[f"{robot_name}_joints"]
    else:
        urdf_model_path = ""
        package_dirs = ""
        urdf_joint_name = ""

    # env
    episode_length_s = 1.0
    decimation = 4

    mode = "train"   # train 或 play，会在 train.py 或 play.py 中指定

    # spaces
    observation_space = 0
    action_space = 1 * 31  # 所有 31 个关节都使用 delta action
    state_space = 0

    early_termination = True
    termination_height = 0.8

    max_payload_mass = 3.0
    robot_mass_range = [1.0, 1.0]

    train_motion_file: str = MISSING  # type: ignore
    reference_body = "waist_pitch_link"
    reset_strategy = "random"  # default, random, random-start
    
    """
    Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**24,
            gpu_total_aggregate_pairs_capacity=2**24,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ROBOT_DICT[robot_name]["model"].replace(prim_path="/World/envs/env_.*/Robot")   # type: ignore

    motion_dir = MOTIONS_DIR
    motion_joint = None
    # motion_file = os.path.join(motion_dir, f"motion_perjoint_all/{ROBOT_DICT[robot_name]['motion_dir']}")
    motion_path = os.path.join(motion_dir, f"motion_amass/{ROBOT_DICT[robot_name]['motion_dir']}")
    train_motion_file = os.path.join(motion_path, "merged_50Hz_31_10_payload.npz")
    # train_motion_file = os.path.join(motion_path, "motor_edited_extend_amass_test_merged_50Hz_full_subset_20.npz")
    # test_motion_file = os.path.join(motion_path, "motor_edited_extend_amass_test_merged_50Hz_full_subset_bwd_20.npz")
    # test_motion_file = os.path.join(motion_path, "test_full.npz")
    # Use a smaller subset of the training motions for play / test to speed up evaluation
    # This file is created from the first 5 motions of merged_50Hz_31_10_payload.npz
    test_motion_file = os.path.join(motion_path, "merged_50Hz_31_10_payload_test_small_5.npz")

    # train_motion_file = os.path.join(motion_path, "motor_edited_extend_amass_train_100Hz.npz")
    # test_motion_file = os.path.join(motion_path, "motor_edited_extend_amass_test_40_100Hz_full.npz")

    # sub environments
    num_sensor_positions = 20
    sensors_positions = [
        {'left_shoulder_pitch_joint': 0.5,
         'right_shoulder_pitch_joint': 0.5,},
        {'left_shoulder_yaw_joint': 0.5,
         'right_shoulder_yaw_joint': 0.5,},
        {'left_shoulder_roll_joint': 0.5,
         'right_shoulder_roll_joint': 0.5,},
        {'left_shoulder_pitch_joint': 0.5,
         'right_shoulder_pitch_joint': 0.5,
         'left_shoulder_yaw_joint': 0.5,
         'right_shoulder_yaw_joint': 0.5,
         'left_shoulder_roll_joint': 0.5,
         'right_shoulder_roll_joint': 0.5,},

        {'left_elbow_pitch_joint': 0.5,
         'right_elbow_pitch_joint': 0.5,},
        {'left_wrist_roll_joint': 0.5,
         'right_wrist_roll_joint': 0.5,},
        {'left_wrist_pitch_joint': 0.5,
         'right_wrist_pitch_joint': 0.5,},
        {'left_wrist_yaw_joint': 0.5,
         'right_wrist_yaw_joint': 0.5,},

        {'left_shoulder_pitch_joint': -0.5,
        'left_shoulder_roll_joint': -0.5,},
        {'left_shoulder_yaw_joint': -0.5,
        'left_elbow_pitch_joint': 0.5,},
        {'left_shoulder_pitch_joint': -0.5,
        'left_wrist_roll_joint': -0.5,},
        {'left_shoulder_yaw_joint': -0.5,
        'left_wrist_pitch_joint': -0.5,},

        {'right_shoulder_pitch_joint': -0.5,
        'right_shoulder_roll_joint': -0.5,},
        {'right_shoulder_yaw_joint': -0.5,
        'right_elbow_pitch_joint': -0.5,},
        {'right_shoulder_pitch_joint': -0.5,
        'right_wrist_roll_joint': -0.5,},
        {'right_shoulder_yaw_joint': -0.5,
        'right_wrist_pitch_joint': -0.5,},
         
        {'left_shoulder_pitch_joint': -0.5,
         'left_elbow_pitch_joint': 0.5,
         'left_wrist_roll_joint': -0.5,
         'right_shoulder_pitch_joint': -0.5,
         'right_elbow_pitch_joint': 0.5,
         'right_wrist_roll_joint': -0.5,},

        {'left_shoulder_pitch_joint': 0.5,
         'left_elbow_pitch_joint': 0.5,
         'left_wrist_roll_joint': 0.5,
         'right_shoulder_pitch_joint': 0.5,
         'right_elbow_pitch_joint': 0.5,
         'right_wrist_roll_joint': 0.5,},

         {'left_shoulder_yaw_joint': -0.5,
         'left_wrist_pitch_joint': -0.5,
         'left_wrist_yaw_joint': -0.5,
         'right_shoulder_yaw_joint': -0.5,
         'right_wrist_pitch_joint': -0.5,
         'right_wrist_yaw_joint': -0.5,},

        {'left_shoulder_yaw_joint': 0.5,
         'left_wrist_pitch_joint': 0.5,
         'left_wrist_yaw_joint': 0.5,
         'right_shoulder_yaw_joint': 0.5,
         'right_wrist_pitch_joint': 0.5,
         'right_wrist_yaw_joint': 0.5,},
    ]
    delta_sensor_position = True
    delta_sensor_value = True

    # 使用 31 DOF 的关节状态做 history：
    # 每一帧 history = joint_pos(31) + joint_vel(31) + joint_target(31) = 93 维
    add_model_history = True
    model_history_length = 4  # 必须与模型配置中的 model_history_length 一致
    model_initial_fill_length = 4
    model_history_dim = 93    # 必须与模型配置中的 model_history_dim 一致

    # 传感器每个位置的特征维度：
    # 这里使用 joint_pos(31) + joint_vel*dt(31) = 62 维，
    # 必须与 env 中 _set_sensor_data / _pre_set_sensor_data 拼接的一致
    sensor_dim = 62
    sensor_decimation = 1

    add_noise = True
    record_sim_mode = False

    
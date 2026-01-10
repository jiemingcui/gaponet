# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from sim2real_assets import H1_2_CFG_WITH_HAND_FIX, H1_2_WITH_HAND_FIX_URDF_PATH, H1_2_CFG_WITH_HAND_FIX_PAYLOAD, H1_2_WITH_HAND_FIX_PAYLOAD_URDF_PATH # type: ignore

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


from .motions.joint_names import ROBOT_JOINT_NAME_DICT_URDF

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

ROBOT_DICT = {
    "h1_2_with_hand_fix": {"model": H1_2_CFG_WITH_HAND_FIX, "motion_dir": "edited_27dof", "urdf_path": H1_2_WITH_HAND_FIX_URDF_PATH},
    "h1_2_with_hand_fix_payload": {"model": H1_2_CFG_WITH_HAND_FIX_PAYLOAD, "motion_dir": "edited_27dof", "urdf_path": H1_2_WITH_HAND_FIX_PAYLOAD_URDF_PATH},
}

@configclass
class HumanoidOperatorEnvCfg(DirectRLEnvCfg):
    robot_name: str = "h1_2_with_hand_fix_payload"
    compute_eq_torque = False

    if 'urdf_path' in ROBOT_DICT[robot_name]:
        urdf_model_path = ROBOT_DICT[robot_name]["urdf_path"]
        package_dirs = os.path.dirname(urdf_model_path)
        urdf_joint_name = ROBOT_JOINT_NAME_DICT_URDF[f"{robot_name}_joints"]
    else:
        urdf_model_path = ""
        package_dirs = ""
        urdf_joint_name = ""

    # env
    episode_length_s = 1.0
    decimation = 4

    mode = "train"   # train or play, specified in train.py or play.py

    # spaces
    observation_space = 0
    action_space = 1 * 10
    state_space = 0

    early_termination = True
    termination_height = 0.8

    max_payload_mass = 3.0
    robot_mass_range = [1.0, 1.0]

    train_motion_file: str = MISSING  # type: ignore
    reference_body = "torso_link"
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
    train_motion_file = os.path.join(motion_path, "test.npz")
    test_motion_file = os.path.join(motion_path, "test.npz")

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

        {'left_elbow_joint': 0.5,
         'right_elbow_joint': 0.5,},
        {'left_wrist_roll_joint': 0.5,
         'right_wrist_roll_joint': 0.5,},
        {'left_wrist_pitch_joint': 0.5,
         'right_wrist_pitch_joint': 0.5,},
        {'left_wrist_yaw_joint': 0.5,
         'right_wrist_yaw_joint': 0.5,},

        {'left_shoulder_pitch_joint': -0.5,
        'left_shoulder_roll_joint': -0.5,},
        {'left_shoulder_yaw_joint': -0.5,
        'left_elbow_joint': 0.5,},
        {'left_shoulder_pitch_joint': -0.5,
        'left_wrist_roll_joint': -0.5,},
        {'left_shoulder_yaw_joint': -0.5,
        'left_wrist_pitch_joint': -0.5,},

        {'right_shoulder_pitch_joint': -0.5,
        'right_shoulder_roll_joint': -0.5,},
        {'right_shoulder_yaw_joint': -0.5,
        'right_elbow_joint': -0.5,},
        {'right_shoulder_pitch_joint': -0.5,
        'right_wrist_roll_joint': -0.5,},
        {'right_shoulder_yaw_joint': -0.5,
        'right_wrist_pitch_joint': -0.5,},
         
        {'left_shoulder_pitch_joint': -0.5,
         'left_elbow_joint': 0.5,
         'left_wrist_roll_joint': -0.5,
         'right_shoulder_pitch_joint': -0.5,
         'right_elbow_joint': 0.5,
         'right_wrist_roll_joint': -0.5,},

        {'left_shoulder_pitch_joint': 0.5,
         'left_elbow_joint': 0.5,
         'left_wrist_roll_joint': 0.5,
         'right_shoulder_pitch_joint': 0.5,
         'right_elbow_joint': 0.5,
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

    add_model_history = True
    model_history_length = 4 # this must match model_history_length in model config
    model_initial_fill_length = 4
    model_history_dim = 30 # this must match model_history_dim in model config

    sensor_dim = 20
    sensor_decimation = 1

    add_noise = True
    record_sim_mode = False

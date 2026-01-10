# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

import gymnasium as gym

from isaaclab_assets import HUMANOID_28_CFG, H1_2_CFG, assets_dir
usd_filename = 'usds/h1_2_tabletop_picknplace.usd'
usd_file_path = os.path.join(assets_dir, usd_filename)

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils

@configclass
class HumanoidTabletopEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env\
    episode_length_s = 10.0
    step_dt = 1/1000
    decimation = 20

    # spaces
    observation_space = 0
    obs_trajectory_length = 0
    action_space = 20

    early_termination = True
    termination_height = 0.8

    task_randomization_scale = 0.05

    reference_body = "torso_link"
    reset_strategy = "random"  # default, random, random-start

    joint_order = [
        "left_hip_yaw_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",

        "L_index_proximal_joint",
        "L_index_intermediate_joint",
        "L_middle_proximal_joint",
        "L_middle_intermediate_joint",
        "L_pinky_proximal_joint",
        "L_pinky_intermediate_joint",
        "L_ring_proximal_joint",
        "L_ring_intermediate_joint",
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_intermediate_joint",
        "L_thumb_distal_joint",

        "R_index_proximal_joint",
        "R_index_intermediate_joint",
        "R_middle_proximal_joint",
        "R_middle_intermediate_joint",
        "R_pinky_proximal_joint",
        "R_pinky_intermediate_joint",
        "R_ring_proximal_joint",
        "R_ring_intermediate_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_intermediate_joint",
        "R_thumb_distal_joint"
    ]

    randomize_pos_xy = [
        [-0.05, 0.05],
        [-0.20, 0.20],
    ]
    randomize_pos_z = [-0.05, 0.0]
    randomize_mass = [1.0, 3.0]

    randomize_cabin_pos = [
        [-0.05, 0.05],
        [-0.20, 0.20],
        [-0.10, 0.10],
    ]

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=step_dt,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True, filter_collisions=True)

    # environment
    environment: UsdFileCfg = UsdFileCfg(
        usd_path=usd_file_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    )

    # payload
    payload: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/task_env/payload",
        spawn=None,
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/task_env/table",
        spawn=None,
    )

    cabin: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/task_env/cabin",
        spawn=None,
    )

    # robot
    robot: ArticulationCfg = H1_2_CFG.replace(prim_path="/World/envs/env_.*/task_env/robot").replace( # type: ignore
            actuators={
                "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    "torso_joint"
                ],
                effort_limit_sim=300,
                velocity_limit_sim=100.0,
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                    "torso_joint": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0,
                    ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0,
                    ".*_knee_joint": 5.0,
                    "torso_joint": 5.0,
                },
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit_sim=100,
                velocity_limit_sim=100.0,
                stiffness={
                    ".*_ankle_pitch_joint": 20.0,
                    ".*_ankle_roll_joint": 20.0
                },
                damping={
                    ".*_ankle_pitch_joint": 4.0,
                    ".*_ankle_roll_joint": 4.0,
                },
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint"
                ],
                effort_limit_sim=300,
                velocity_limit_sim=100.0,
                stiffness={
                    ".*_shoulder_pitch_joint": 100.0,
                    ".*_shoulder_roll_joint": 100.0,
                    ".*_shoulder_yaw_joint": 100.0,
                    ".*_elbow_joint": 100.0,
                    ".*_wrist_roll_joint": 100.0,
                    ".*_wrist_pitch_joint": 100.0,
                    ".*_wrist_yaw_joint": 100.0,
                },
                damping={
                    ".*_shoulder_pitch_joint": 2.0,
                    ".*_shoulder_roll_joint": 2.0,
                    ".*_shoulder_yaw_joint": 2.0,
                    ".*_elbow_joint": 2.0,
                    ".*_wrist_roll_joint": 2.0,
                    ".*_wrist_pitch_joint": 2.0,
                    ".*_wrist_yaw_joint": 2.0,
                },
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["L_.*", "R_.*"],
                velocity_limit_sim=100.0,
                stiffness={
                    ".*proximal.*": 100.0,
                    ".*intermediate.*": 200.0,
                    ".*distal.*": 200.0,
                },
                damping={
                    ".*proximal.*": 2.0,
                    ".*intermediate.*": 4.0,
                    ".*distal.*": 4.0,
                },
                effort_limit_sim={
                    ".*proximal.*": 1000.0,
                    ".*intermediate.*": 1000.0,
                    ".*distal.*": 1000.0,
                },
            ),
        },
    ).replace(
        spawn=None
    )

    markers = VisualizationMarkersCfg(
                prim_path="/World/testMarkers",
                markers={
                    "left_hand_marker": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), emissive_color=(1.0, 0.0, 0.0), metallic=0.5, roughness=0.5, opacity=1.0),
                    ),
                    "right_hand_marker": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 1.0, 0.0), metallic=0.5, roughness=0.5, opacity=1.0),
                    ),
                    "target_marker": sim_utils.ConeCfg(
                        radius=0.02,
                        height=0.04,
                        axis='Z',
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), emissive_color=(0.0, 0.0, 1.0), metallic=0.5, roughness=0.5, opacity=1.0),
                    ),
                }
            )

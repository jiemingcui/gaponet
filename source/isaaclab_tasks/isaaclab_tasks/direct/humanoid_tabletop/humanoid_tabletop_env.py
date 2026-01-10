# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate

from .humanoid_tabletop_env_cfg import HumanoidTabletopEnvCfg
from isaaclab.envs.mdp.actions.command_to_actions import IKEEFToActions
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from typing import Dict, Tuple, Any

class HumanoidTabletopEnv(DirectRLEnv):
    cfg: HumanoidTabletopEnvCfg

    def __init__(self, cfg: HumanoidTabletopEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def joint_converter(correct_joint_name: Sequence[str], joint_name: Sequence[str]) -> Tuple[Sequence[int], Sequence[int]]:
            remap_s2c = []
            for i, name in enumerate(joint_name):
                remap_s2c.append(correct_joint_name.index(name))
            remap_c2s = []
            for i, name in enumerate(correct_joint_name):
                remap_c2s.append(joint_name.index(name))
            return remap_s2c, remap_c2s
        self.S2C, self.C2S = joint_converter(self.cfg.joint_order, self.robot.data.joint_names)

        self.num_joints = self.robot.num_joints
        self.num_body_joints = self.cfg.joint_order.index("right_wrist_yaw_joint") + 1
        self.num_lower_joints = self.cfg.joint_order.index("torso_joint") + 1
        self.num_single_hand_joints = (self.num_joints - self.num_body_joints) // 2

        self.proximal_thumb_joints = [i for i, name in enumerate(self.cfg.joint_order) if "proximal" in name and "thumb" in name]
        self.proximal_hand_joints = [i for i, name in enumerate(self.cfg.joint_order) if "proximal" in name and ('L' in name or 'R' in name)]
        self.intermediate_thumb_joints = [i for i, name in enumerate(self.cfg.joint_order) if not "proximal" in name and "thumb" in name]
        self.intermediate_hand_joints = [i for i, name in enumerate(self.cfg.joint_order) if not "proximal" in name and ('L' in name or 'R' in name)]
        
        self.s_left_hand_joints = [self.C2S[i] for i in range(self.num_body_joints, self.num_body_joints+self.num_single_hand_joints)]
        self.s_right_hand_joints = [self.C2S[i] for i in range(self.num_body_joints+self.num_single_hand_joints, self.num_joints)]
        self.s_proximal_thumb_joints = [self.C2S[i] for i in self.proximal_thumb_joints]
        self.s_proximal_hand_joints = [self.C2S[i] for i in self.proximal_hand_joints]
        self.s_intermediate_thumb_joints = [self.C2S[i] for i in self.intermediate_thumb_joints]

        self.s_lower_joints = [self.C2S[i] for i in range(self.num_lower_joints)]
        self.s_upper_joints = [self.C2S[i] for i in range(self.num_lower_joints, self.num_body_joints)]

        self.time_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.joint_actions = torch.zeros(self.num_envs, self.num_joints, device=self.device)

        self.joint_lower_limit = self.robot.data.soft_joint_pos_limits[:, :, 0]
        self.joint_upper_limit = self.robot.data.soft_joint_pos_limits[:, :, 1]

        self.left_hand_lower_limit = self.joint_lower_limit[:, self.C2S[self.num_body_joints:self.num_body_joints+self.num_single_hand_joints]]
        self.left_hand_upper_limit = self.joint_upper_limit[:, self.C2S[self.num_body_joints:self.num_body_joints+self.num_single_hand_joints]]
        self.right_hand_lower_limit = self.joint_lower_limit[:, self.C2S[self.num_body_joints+self.num_single_hand_joints:]]
        self.right_hand_upper_limit = self.joint_upper_limit[:, self.C2S[self.num_body_joints+self.num_single_hand_joints:]]

        self.initial_payload_pos = self.payload.data.body_pos_w[:, 0].clone()
        self.initial_payload_quat = self.payload.data.body_quat_w[:, 0].clone()
        self.initial_table_pos = self.table.data.root_pos_w[:].clone()
        self.initial_table_quat = self.table.data.root_quat_w[:].clone()
        self.initial_cabin_pos = self.cabin.data.root_pos_w[:].clone()
        self.initial_cabin_quat = self.cabin.data.root_quat_w[:].clone()

        # retrieve material buffer from the physics simulation
        material_samples = torch.zeros(self.num_envs, 1, 3, device="cpu")
        material_samples[:, :, 0:2] = 100.0

        materials = self.payload.root_physx_view.get_material_properties()
        materials[:] = material_samples[:]
        # apply to simulation
        self.payload.root_physx_view.set_material_properties(materials, self.payload._ALL_INDICES.cpu()) # type: ignore

        materials = self.robot.root_physx_view.get_material_properties()
        materials[:] = material_samples[:]
        # apply to simulation
        self.robot.root_physx_view.set_material_properties(materials, self.robot._ALL_INDICES.cpu()) # type: ignore

        self.initialize_ik_solvers()

        ##########################
        # Task Related Variables #
        ##########################
        self.stable_payload_pos = self.payload.data.root_pos_w.clone()
        self.stable_payload_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.trajectory_buffer = torch.zeros(self.num_envs, self.cfg.obs_trajectory_length * 256, device=self.device)
        self.trajectory_length = self.cfg.obs_trajectory_length

    def _setup_scene(self):
        self.cfg.environment.func("/World/envs/env_.*/task_env", self.cfg.environment)
        self.robot = Articulation(self.cfg.robot)
        self.payload = RigidObject(self.cfg.payload)
        self.table = RigidObject(self.cfg.table)
        self.cabin = RigidObject(self.cfg.cabin)

        # indices 0, 1 for L/R hands and 2 for target
        self.markers = VisualizationMarkers(self.cfg.markers)
        # add ground plane
        # spawn_ground_plane(
        #     prim_path="/World/ground",
        #     cfg=GroundPlaneCfg(
        #         physics_material=sim_utils.RigidBodyMaterialCfg(
        #             static_friction=1.0,
        #             dynamic_friction=1.0,
        #             restitution=0.0,
        #         ),
        #     ),
        # )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation and rigid object to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["payload"] = self.payload
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["cabin"] = self.cabin

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.clone()
        
        gripper_action = actions[:, -2:].clamp(0, 1)
        thumb_action = actions[:, -6:-2]
        body_action = actions[:, :-6]
        self.actions[:] = actions

        self.joint_actions[:, self.s_lower_joints] = 0.
        self.joint_actions[:, self.s_upper_joints] = body_action
        self.joint_actions[:, self.s_left_hand_joints] = gripper_action[:, 0, None] * (self.left_hand_upper_limit - self.left_hand_lower_limit) + self.left_hand_lower_limit
        self.joint_actions[:, self.s_right_hand_joints] = gripper_action[:, 1, None] * (self.right_hand_upper_limit - self.right_hand_lower_limit) + self.right_hand_lower_limit
        self.joint_actions[:, self.s_intermediate_thumb_joints] = self.joint_lower_limit[:, self.s_intermediate_thumb_joints] + self.joint_upper_limit[:, self.s_intermediate_thumb_joints] - self.joint_actions[:, self.s_intermediate_thumb_joints]
        self.joint_actions[:, self.s_proximal_thumb_joints] = thumb_action

    def _apply_action(self):
        self.robot.set_joint_position_target(self.joint_actions)

    def _get_observations(self) -> dict:
        rel_payload_pos = self.payload.data.root_pos_w - self.robot.data.body_link_pos_w[:, 0]
        rel_target_pos = self.target_pos - self.robot.data.body_link_pos_w[:, 0]

        joint_pos = self.robot.data.joint_pos[:, self.s_upper_joints]
        joint_vel = self.robot.data.joint_vel[:, self.s_upper_joints]
        last_action = self.actions

        payload_mass = self.payload.root_physx_view.get_masses().to(self.device)

        step_obs = torch.cat([rel_payload_pos, rel_target_pos, joint_pos, joint_vel, last_action, payload_mass], dim=-1)
        if self.cfg.obs_trajectory_length > 0:
            history_obs = torch.cat([rel_payload_pos, joint_pos, joint_vel], dim=-1)
            history_obs_size = history_obs.shape[-1]
            self.trajectory_buffer[:, history_obs_size:] = self.trajectory_buffer[:, :-history_obs_size].clone()
            self.trajectory_buffer[:, :history_obs_size] = history_obs.clone()

            return {"policy": torch.cat([step_obs, self.trajectory_buffer[:, :self.trajectory_length * history_obs_size]], dim=-1).clone()}
        else:
            return {"policy": step_obs.clone()}

    def _get_rewards(self) -> torch.Tensor:
        ''' This Environment is not used for RL training. '''
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= (self.max_episode_length - 1)

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.early_termination:
            terminated |= (self.payload.data.body_pos_w[:, 0, 2] < self.cfg.termination_height)

        # Update time_indices
        self.time_indices += 1

        self.stable_payload_pos[~self.stable_payload_flag] = self.payload.data.root_pos_w[~self.stable_payload_flag]
        self.stable_payload_flag |= (torch.norm(self.payload.data.root_vel_w[:, :3], dim=-1) < 0.01)
        
        return terminated, truncated

    def _get_ready(self) -> torch.Tensor:
        return self.stable_payload_flag
    
    def _get_success(self) -> torch.Tensor:
        return (torch.norm(self.payload.data.root_pos_w[:, :2] - self.target_pos[:, :2], dim=-1) < 0.07) \
            & (torch.norm(self.payload.data.root_vel_w[:], dim=-1) < 0.01) \
            & (torch.norm(self.payload.data.root_pos_w[:, 2] - self.target_pos[:, 2], dim=-1) < 0.10)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES # type: ignore

        self.trajectory_buffer[env_ids, :] = 0

        # Reset payload position
        self.payload.reset(env_ids) # type: ignore
        self.apply_task_randomization(env_ids)

        # Reset robot state
        self.robot.reset(env_ids) # type: ignore
        self.robot.write_joint_position_to_sim(self.robot.data.joint_pos * 0, env_ids=env_ids) # type: ignore
        self.robot.write_joint_velocity_to_sim(self.robot.data.joint_vel * 0, env_ids=env_ids) # type: ignore

        # Reset ik solvers
        if 'left_ik_solver' in self.__dict__:
            self.left_ik_solver.reset()
            self.right_ik_solver.reset()
            self.left_thumb_solver.reset()
            self.right_thumb_solver.reset()

        # Reset environment
        super()._reset_idx(env_ids) # type: ignore

    def randomize_target_pos(self, payload_pos: torch.Tensor, env_ids: torch.Tensor):
        self.target_pos[env_ids, 0] = self.initial_cabin_pos[env_ids, 0] + (self.cfg.task_randomization_scale) * (torch.rand(len(env_ids), device=self.device) * (self.cfg.randomize_cabin_pos[0][1] - self.cfg.randomize_cabin_pos[0][0]) + self.cfg.randomize_cabin_pos[0][0])
        self.target_pos[env_ids, 1] = self.initial_cabin_pos[env_ids, 1] + (self.cfg.task_randomization_scale) * (torch.rand(len(env_ids), device=self.device) * (self.cfg.randomize_cabin_pos[1][1] - self.cfg.randomize_cabin_pos[1][0]) + self.cfg.randomize_cabin_pos[1][0])
        self.target_pos[env_ids, 2] = self.initial_cabin_pos[env_ids, 2] + (self.cfg.task_randomization_scale) * (torch.rand(len(env_ids), device=self.device) * (self.cfg.randomize_cabin_pos[2][1] - self.cfg.randomize_cabin_pos[2][0]) + self.cfg.randomize_cabin_pos[2][0])
        self.cabin.write_root_pose_to_sim(torch.cat([self.target_pos[env_ids, :3], self.initial_cabin_quat[env_ids, :]], dim=-1), env_ids) # type: ignore

    def apply_task_randomization(self, env_ids: torch.Tensor):
        # Randomize payload position
        root_state = torch.zeros(len(env_ids), 13, device=self.device)
        root_state[:, 0] = self.initial_payload_pos[env_ids, 0] + (self.cfg.task_randomization_scale) * (torch.rand(len(env_ids), device=self.device) * (self.cfg.randomize_pos_xy[0][1] - self.cfg.randomize_pos_xy[0][0]) + self.cfg.randomize_pos_xy[0][0])
        root_state[:, 1] = self.initial_payload_pos[env_ids, 1] + (self.cfg.task_randomization_scale) * (torch.rand(len(env_ids), device=self.device) * (self.cfg.randomize_pos_xy[1][1] - self.cfg.randomize_pos_xy[1][0]) + self.cfg.randomize_pos_xy[1][0])
        root_state[:, 2] = self.initial_payload_pos[env_ids, 2]
        root_state[:, 3] = 1.
        self.payload.write_root_state_to_sim(root_state, env_ids) # type: ignore
        self.stable_payload_flag[env_ids] = False

        root_pose = torch.zeros(len(env_ids), 7, device=self.device)
        root_pose[:, :3] = self.initial_table_pos[env_ids, :]
        root_pose[:, 2] += (self.cfg.task_randomization_scale) * (torch.rand(len(env_ids), device=self.device) * (self.cfg.randomize_pos_z[1] - self.cfg.randomize_pos_z[0]) + self.cfg.randomize_pos_z[0])
        root_pose[:, 3:] = self.initial_table_quat[env_ids, :]
        self.table.write_root_pose_to_sim(root_pose, env_ids) # type: ignore

        # Randomize target position
        self.randomize_target_pos(self.payload.data.root_pos_w, env_ids)

        # Randomize payload mass
        # CPU is necessary for this operation
        masses = torch.rand(len(env_ids), 1, device="cpu") * (self.cfg.randomize_mass[1] - self.cfg.randomize_mass[0]) + self.cfg.randomize_mass[0]
        self.payload.root_physx_view.set_masses(masses, env_ids.cpu()) # type: ignore

    def initialize_ik_solvers(self):
        left_solver_cfg = DifferentialInverseKinematicsActionCfg(
            joint_names=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
            body_name="left_wrist_yaw_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False, 
                ik_method="pinv",
                ik_params={
                    "k_val": 0.6,
                }
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0., 0., 0.),
            )
        )
        self.left_ik_solver = IKEEFToActions(left_solver_cfg, self, self.robot, None, None)

        right_solver_cfg = DifferentialInverseKinematicsActionCfg(
            joint_names=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
            body_name="right_wrist_yaw_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False, 
                ik_method="pinv",
                ik_params={
                    "k_val": 0.6,
                }
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0., 0., 0.),
            )
        )
        self.right_ik_solver = IKEEFToActions(right_solver_cfg, self, self.robot, None, None)

        left_thumb_solver_cfg = DifferentialInverseKinematicsActionCfg(
            joint_names=["L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint"],
            body_name="L_thumb_distal",
            controller=DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=False, 
                ik_method="dls")
        )
        self.left_thumb_solver = IKEEFToActions(left_thumb_solver_cfg, self, self.robot, None, None, "left_wrist_yaw_link")

        right_thumb_solver_cfg = DifferentialInverseKinematicsActionCfg(
            joint_names=["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint"],
            body_name="R_thumb_distal",
            controller=DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=False, 
                ik_method="dls")
        )
        self.right_thumb_solver = IKEEFToActions(right_thumb_solver_cfg, self, self.robot, None, None, "right_wrist_yaw_link")

    def commands_to_action(self, commands: Dict[str, torch.Tensor]) -> torch.Tensor:
        head_pose = commands["head"].view(self.num_envs, 7)
        left_palm = commands["left_hand"][0].view(self.num_envs, 7)
        right_palm = commands["right_hand"][0].view(self.num_envs, 7)

        valid_pose = torch.all(~(head_pose[:, :3] == 0)).item()
        if not valid_pose:
            left_idx = self.robot.body_names.index("left_wrist_yaw_link")
            right_idx = self.robot.body_names.index("right_wrist_yaw_link")

            left_palm = torch.cat([self.robot.data.body_link_pos_w[:, left_idx, :], self.robot.data.body_link_quat_w[:, left_idx, :]], dim=-1)
            right_palm = torch.cat([self.robot.data.body_link_pos_w[:, right_idx, :], self.robot.data.body_link_quat_w[:, right_idx, :]], dim=-1)

        rel_left_palm_pose = math_utils.subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w, left_palm[:, :3], left_palm[:, 3:])
        rel_right_palm_pose = math_utils.subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w, right_palm[:, :3], right_palm[:, 3:])
        rel_left_palm_pose = torch.cat([rel_left_palm_pose[0], rel_left_palm_pose[1]], dim=-1)
        rel_right_palm_pose = torch.cat([rel_right_palm_pose[0], rel_right_palm_pose[1]], dim=-1)

        self.last_left_palm_pose = rel_left_palm_pose.clone()
        self.last_right_palm_pose = rel_right_palm_pose.clone()
        
        self.markers.visualize(translations=torch.cat([left_palm[:, :3], right_palm[:, :3], self.target_pos * self.stable_payload_flag[:, None]], dim=0), marker_indices=[0, 1, 2])

        # Solve IK
        self.left_ik_solver.process_actions(rel_left_palm_pose[:, :7])
        left_joint_target, left_ids = self.left_ik_solver.compute_actions()

        self.right_ik_solver.process_actions(rel_right_palm_pose[:, :7])
        right_joint_target, right_ids = self.right_ik_solver.compute_actions()

        left_thumb_pose = commands["left_hand"][4].view(self.num_envs, 7)
        right_thumb_pose = commands["right_hand"][4].view(self.num_envs, 7)
        rel_left_thumb_pose = math_utils.subtract_frame_transforms(left_palm[:, :3], left_palm[:, 3:], left_thumb_pose[:, :3], left_thumb_pose[:, 3:])
        rel_right_thumb_pose = math_utils.subtract_frame_transforms(right_palm[:, :3], right_palm[:, 3:], right_thumb_pose[:, :3], right_thumb_pose[:, 3:])
        
        self.left_thumb_solver.process_actions(rel_left_thumb_pose[0])
        left_thumb_joint_target, left_thumb_ids = self.left_thumb_solver.compute_actions()

        self.right_thumb_solver.process_actions(rel_right_thumb_pose[0])
        right_thumb_joint_target, right_thumb_ids = self.right_thumb_solver.compute_actions()

        actions = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        actions[:, left_ids] = left_joint_target
        actions[:, right_ids] = right_joint_target
        actions[:, left_thumb_ids] = left_thumb_joint_target
        actions[:, right_thumb_ids] = right_thumb_joint_target
        body_actions = actions[:, self.C2S[self.num_lower_joints:self.num_body_joints]]
        thumb_actions = actions[:, [self.C2S[self.proximal_thumb_joints[0]], self.C2S[self.proximal_thumb_joints[1]], self.C2S[self.proximal_thumb_joints[2]], self.C2S[self.proximal_thumb_joints[3]]]]

        gripper_actions = commands["gripper_command"].view(self.num_envs, 2)
        return torch.cat([body_actions, thumb_actions, gripper_actions], dim=-1)
    
    def get_state(self, is_relative: bool = True) -> dict[str, Any]:
        return {
            "env_state": self.scene.get_state(is_relative=is_relative),
            "env_property": {
                "payload_mass": self.payload.root_physx_view.get_masses().clone(),
            }
        }
    
    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> None:
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset environment state
        env_state = state["env_state"]
        self._reset_idx(env_ids) # type: ignore
        self.scene.reset_to(env_state, env_ids, is_relative=is_relative) # type: ignore

        # reset environment properties
        env_property = state["env_property"]
        if "payload_mass" in env_property:
            self.payload.root_physx_view.set_masses(env_property["payload_mass"], env_ids.cpu()) # type: ignore


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import os
from datetime import datetime

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate

# import pinocchio as pin
import pytorch_kinematics as pk
from copy import deepcopy

from .humanoid_operator_env_cfg import HumanoidOperatorEnvCfg
from .motions import MotionLoaderMotor
from .operator_helper import get_sensor_positions, set_masses, reset_masses

from typing import Dict, Tuple

class HumanoidOperatorEnv(DirectRLEnv):
    cfg: HumanoidOperatorEnvCfg

    def __init__(self, cfg: HumanoidOperatorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ALL_INDICES = torch.arange(self.num_envs, device=self.device)

        self.num_sensor_positions = self.cfg.num_sensor_positions
        self.num_sub_environments = self.num_envs // self.num_sensor_positions
        print("================== num_sub_environments: ", self.num_sub_environments, "==================")
        print("================== num_envs: ", self.num_envs, "==================")
        print("================== num_sensor_positions: ", self.num_sensor_positions, "==================")
        if not self.cfg.mode == "play":
            assert self.num_envs % self.num_sensor_positions == 0, "num_envs must be divisible by num_sensor_positions"
        

        # action offset and scale
        self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[:, :, 0]
        self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[:, :, 1]
        self.joint_vel_limits = self.robot.data.soft_joint_vel_limits[:, :]
        
        self.mode = self.cfg.mode

        # load motion
        self._motion_loader = MotionLoaderMotor(motion_file=self.cfg.train_motion_file if self.mode == "train" else self.cfg.test_motion_file, device=self.device, mode=self.mode, robot_name=self.cfg.robot_name)  # type: ignore
        self.num_dofs = self._motion_loader.num_dofs

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)

        # Initialize motion and time indices for environments
        self.motion_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.num_actions: int = self.cfg.action_space # type: ignore
        self.last_delta_action = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.delta_action = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.apply_action = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.step_velocity = torch.zeros((self.num_envs, self.num_dofs), device=self.device)

        self.sub_env_sensor_data = torch.zeros((self.num_sub_environments, self.num_sensor_positions, self.cfg.sensor_dim), device=self.device)
        self.sensor_data = torch.zeros((self.num_envs, self.num_sensor_positions, self.cfg.sensor_dim), device=self.device)
        self._init_sensor_positions()

        self.wrist_payload_mass = torch.zeros((self.num_envs, 1), device=self.device) + 0.001
        self._set_wrist_payload_mass()

        self.hand_payload_mass = torch.zeros((self.num_envs, 2), device=self.device) + 0.001
        self._set_hand_payload_mass()

        self.robot_mass = self.robot.data.default_mass.to(device=self.device)

        self.delta_action_joint_indices = self._motion_loader.joint_sequence_index

        self.add_model_history = self.cfg.add_model_history
        self.model_history = torch.zeros((self.num_envs, self.cfg.model_history_length, self.cfg.model_history_dim), device=self.device)

        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.total_save_times = 0

        # minimum number of runs to aggregate results (default 6)
        self.min_runs = getattr(self.cfg, 'min_runs', 4)

        if self.mode == "play":
            self.add_noise = False
            self.init_play_environments()
        else:
            self.add_noise = self.cfg.add_noise

        if self.cfg.urdf_model_path != "":
            with open(self.cfg.urdf_model_path) as f:
                urdf_model = f.read().encode('utf-8')
            self.robot_chain = pk.build_chain_from_urdf(urdf_model).to(device=torch.device(self.device))
            self.chain_joint_names = self.robot_chain.get_joint_parameter_names()
            self.robot_joint_names = self.robot.data.joint_names
            self.joint_usd_to_fk_chain = [self.robot_joint_names.index(name) for name in self.chain_joint_names]

        self.joint_urdf_to_joint_usd = [self.cfg.urdf_joint_name.index(name) for name in self._motion_loader.dof_names]  # Index of joint_usd in joint_urdf
        self.joint_usd_to_joint_urdf = [self._motion_loader.dof_names.index(name) for name in self.cfg.urdf_joint_name]  # Index of joint_urdf in joint_usd
        if self.cfg.compute_eq_torque:
            # Pinocchio model for computing equivalent torque
            self.pin_model, _, _ = pin.buildModelsFromUrdf(self.cfg.urdf_model_path, package_dirs=self.cfg.package_dirs) # type: ignore
            self.pin_data = self.pin_model.createData()

            inertias = []
            for joint_name in self.cfg.urdf_joint_name:
                joint_id = self.pin_model.getJointId(joint_name)
                inertia = self.pin_model.inertias[joint_id]
                joint = self.pin_model.joints[joint_id]
                
                # Get joint axis
                if joint.shortname() == "JointModelRZ":
                    axis = np.array([0, 0, 1])
                elif joint.shortname() == "JointModelRY":
                    axis = np.array([0, 1, 0])
                elif joint.shortname() == "JointModelRX":
                    axis = np.array([1, 0, 0])
                
                axis = axis / np.linalg.norm(axis)  # Normalize
                I_axis = axis @ inertia.inertia @ axis
                inertias.append(I_axis)
            self.inertias = np.array(inertias)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        self.payload1 = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_.*/Robot/payload1",
            )
        )
        self.payload2 = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_.*/Robot/payload2",
            )
        )
        self.payload3 = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_.*/Robot/payload3",
            )
        )
        self.payload4 = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_.*/Robot/payload4",
            )
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["payload1"] = self.payload1
        self.scene.rigid_objects["payload2"] = self.payload2
        self.scene.rigid_objects["payload3"] = self.payload3
        self.scene.rigid_objects["payload4"] = self.payload4
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def solve_fk(self, dof_positions: torch.Tensor):
        ret = self.robot_chain.forward_kinematics(dof_positions[:, self.joint_usd_to_fk_chain])
        eff_names = ["L_hand_base_link", "R_hand_base_link"]

        trans = []
        for eff in eff_names:
            trans.append(ret[eff].get_matrix()[:, :3, 3])
        return trans

    def init_play_environments(self, reload_motion: bool = False):
        self.obs_history = []
        self.reset_nums = 0
        self.saved_nums = 0

        if reload_motion:
            self._motion_loader = MotionLoaderMotor(motion_file=self.cfg.train_motion_file if self.mode == "train" else self.cfg.test_motion_file, device=self.device, mode=self.mode, robot_name=self.cfg.robot_name)  # type: ignore

        self.motion_indices[:], self.time_indices[:] = self._motion_loader.sample_indices(self.num_envs)
        self.robot.write_joint_state_to_sim(self._motion_loader.dof_positions[self.motion_indices, self.time_indices], self._motion_loader.dof_velocities[self.motion_indices, self.time_indices])
        
        if self._motion_loader.hand_marker is not None:
            self.hand_payload_mass[:] = 0.001
            self.hand_payload_mass[self._ALL_INDICES, self._motion_loader.hand_marker[self.motion_indices]] = self._motion_loader.payload_sequence[self.motion_indices].view(-1) + 0.001 # type: ignore
            self.wrist_payload_mass[:] = 0.001
        else:
            self.hand_payload_mass[:] = 0.001
            self.wrist_payload_mass[:] = self._motion_loader.payload_sequence[self.motion_indices] + 0.001

        self._set_wrist_payload_mass()
        self._set_hand_payload_mass()

        self.play_data_history = {}
        self.play_data = [[] for _ in range(self.num_envs)]
        # Track how many times each motion has been run
        self.motion_run_counts = torch.zeros(self._motion_loader.motion_num, dtype=torch.long, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor): 
        self.actions = actions.clone()    # shape: (num_envs, 10)

    def _apply_action(self):
        delta_action = torch.zeros((self.num_envs, self.num_dofs), device=self.device)  # shape: (num_envs, num_dofs)
        delta_action[:, self._motion_loader.joint_sequence_index] = self.actions * (not self.cfg.record_sim_mode)

        # get target position
        dof_target_pos = self._motion_loader.dof_target_pos[self.motion_indices, self.time_indices]

        self.apply_action = (dof_target_pos + delta_action).clone()
        self.robot.set_joint_position_target(self.apply_action)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        died = torch.zeros_like(time_out)
        
        # update time_indices
        self.time_indices += 1

        if self.mode == "play":
            # print("================== sim_dof_positions is not None: ", self._motion_loader.sim_dof_positions is not None, "==================")
            if self._motion_loader.sim_dof_positions is not None:
                sim_left_hand_pos, sim_right_hand_pos = self.solve_fk(self._motion_loader.sim_dof_positions[self.motion_indices, self.time_indices])
            left_hand_pos, right_hand_pos = self.solve_fk(self.robot.data.joint_pos)
            target_left_hand_pos, target_right_hand_pos = self.solve_fk(self._motion_loader.dof_positions[self.motion_indices, self.time_indices])
            for i in range(self.num_envs):
                if self.cfg.record_sim_mode:
                    self.play_data[i].append({
                        "delta_joint_pos": self.robot.data.joint_pos[i:i+1, :].clone().cpu().numpy(),
                    })
                    continue
                self.play_data[i].append({
                    "delta_joint_pos": self.robot.data.joint_pos[i:i+1, self._motion_loader.joint_sequence_index].clone().cpu().numpy(),
                    "real_joint_pos": self._motion_loader.dof_positions[self.motion_indices[i:i+1], self.time_indices[i:i+1]][:, self._motion_loader.joint_sequence_index].clone().cpu().numpy(),
                    "joint_target_pos": self._motion_loader.dof_target_pos[self.motion_indices[i:i+1], self.time_indices[i:i+1]][:, self._motion_loader.joint_sequence_index].clone().cpu().numpy(),
                    "joint_pos_diff": self.robot.data.joint_pos[i:i+1, self._motion_loader.joint_sequence_wo_wrist].clone().cpu().numpy() - self._motion_loader.dof_positions[self.motion_indices[i:i+1], self.time_indices[i:i+1]][:, self._motion_loader.joint_sequence_wo_wrist].clone().cpu().numpy(),
                    "payload_mass": self._motion_loader.payload_sequence[self.motion_indices[i:i+1]].clone().cpu().item(),
                    "hand_marker": self._motion_loader.hand_marker[self.motion_indices[i:i+1]].clone().cpu().item() if self._motion_loader.hand_marker is not None else None,
                    
                    "left_hand_pos": left_hand_pos[i:i+1].clone().cpu().numpy(),
                    "right_hand_pos": right_hand_pos[i:i+1].clone().cpu().numpy(),
                    "target_left_hand_pos": target_left_hand_pos[i:i+1].clone().cpu().numpy(),
                    "target_right_hand_pos": target_right_hand_pos[i:i+1].clone().cpu().numpy(),
                })
                if self._motion_loader.sim_dof_positions is not None:
                    self.play_data[i][-1]['sim_joint_pos'] = self._motion_loader.sim_dof_positions[self.motion_indices[i:i+1], self.time_indices[i:i+1]][:, self._motion_loader.joint_sequence_index].clone().cpu().numpy()
                    self.play_data[i][-1]['sim_joint_pos_diff'] = self._motion_loader.dof_positions[self.motion_indices[i:i+1], self.time_indices[i:i+1]][:, self._motion_loader.joint_sequence_wo_wrist].clone().cpu().numpy() - self._motion_loader.sim_dof_positions[self.motion_indices[i:i+1], self.time_indices[i:i+1]][:, self._motion_loader.joint_sequence_wo_wrist].clone().cpu().numpy()
                    self.play_data[i][-1]['sim_left_hand_pos'] = sim_left_hand_pos[i:i+1].clone().cpu().numpy()
                    self.play_data[i][-1]['sim_right_hand_pos'] = sim_right_hand_pos[i:i+1].clone().cpu().numpy()
                if self._motion_loader.joint_index is not None:
                    self.play_data[i][-1]['single_joint_index'] = self._motion_loader.joint_index[self.motion_indices[i:i+1]].clone().cpu().numpy()

        # check if need to reset (time_indices exceeds motion_len)
        motion_done = self.time_indices >= self._motion_loader.motion_len[self.motion_indices] - 1
        return died, motion_done

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        env_ids: The environment IDs to reset. If None, reset all environments.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES  # type: ignore

        if self.mode == "play" and len(env_ids) > 0:
            # Ensure motion_run_counts is initialized
            if not hasattr(self, 'motion_run_counts'):
                self.motion_run_counts = torch.zeros(self._motion_loader.motion_num, dtype=torch.long, device=self.device)
            # Record reset count
            self.reset_nums += 1
            for env_id in env_ids.tolist():
                if len(self.play_data[env_id]) > 0:
                    motion_idx = self.motion_indices[env_id].item()
                    motion_idx_str = str(motion_idx)
                    if motion_idx_str not in self.play_data_history:
                        self.play_data_history[motion_idx_str] = []
                    self.play_data_history[motion_idx_str].append(deepcopy(self.play_data[env_id]))
                    self.play_data[env_id] = []
                    # Increment run count for this motion
                    self.motion_run_counts[motion_idx] += 1
            # progress print
            counts = {i: self.motion_run_counts[i].item() for i in range(self._motion_loader.motion_num)}
            print("Motion run counts:", counts)
            # stop when each motion has exactly min_runs (or more, but we'll prevent sampling more)
            if all(self.motion_run_counts[i].item() >= self.min_runs for i in range(self._motion_loader.motion_num)):
                self.close()
                exit()

        # Sample motion indices
        # NOTE: In play mode, we ensure each motion runs exactly min_runs times.
        # In train mode, we use random sampling as before (no changes to training behavior).
        if self.mode == "play":
            # Ensure motion_run_counts is initialized
            if not hasattr(self, 'motion_run_counts'):
                self.motion_run_counts = torch.zeros(self._motion_loader.motion_num, dtype=torch.long, device=self.device)
            # Get available motions (those that haven't reached min_runs)
            available_motions = torch.where(self.motion_run_counts < self.min_runs)[0]
            if len(available_motions) == 0:
                # All motions have reached min_runs, should have exited above, but just in case
                self.close()
                exit()
            
            # Sample from available motions
            num_to_sample = len(env_ids)
            if len(available_motions) >= num_to_sample:
                # Randomly sample from available motions
                sampled_indices = torch.randperm(len(available_motions), device=self.device)[:num_to_sample]
                motion_indices = available_motions[sampled_indices]
            else:
                # Not enough available motions, sample with replacement but prioritize under-sampled motions
                # Calculate how many more runs each motion needs
                needed_runs = self.min_runs - self.motion_run_counts[available_motions]
                # Create probability distribution weighted by needed runs
                probs = needed_runs.float() / needed_runs.sum().float()
                sampled_indices = torch.multinomial(probs, num_to_sample, replacement=True)
                motion_indices = available_motions[sampled_indices]
            
            # Sample time indices (always start from 0 in play mode)
            time_indices = torch.zeros(num_to_sample, dtype=torch.long, device=self.device)
            self.motion_indices[env_ids] = motion_indices
            self.time_indices[env_ids] = time_indices
        else:
            # Train mode: use original random sampling logic (unchanged)
            # Randomly reset motion_indices and time_indices, shape: [(len(env_ids), ), (len(env_ids), )]
            self.motion_indices[env_ids], self.time_indices[env_ids] = self._motion_loader.sample_indices(len(env_ids))   

        motion_indices = self.motion_indices[env_ids]
        time_indices = self.time_indices[env_ids]

        self.last_delta_action[env_ids] *= 0

        self.robot.reset(env_ids)  # type: ignore
        super()._reset_idx(env_ids) # type: ignore

        self.robot.write_joint_state_to_sim(self._motion_loader.dof_positions[motion_indices, time_indices], self._motion_loader.dof_velocities[motion_indices, time_indices], None, env_ids)   # type: ignore shape: (10,)
        if self._motion_loader.hand_marker is not None:
            self.hand_payload_mass[env_ids] = 0.001
            self.wrist_payload_mass[env_ids] = 0.001
            self.hand_payload_mass[env_ids, self._motion_loader.hand_marker[motion_indices]] = self._motion_loader.payload_sequence[motion_indices].view(-1) + 0.001 # type: ignore
            
        else:
            self.hand_payload_mass[env_ids] = 0.001
            self.wrist_payload_mass[env_ids] = self._motion_loader.payload_sequence[motion_indices] + 0.001
        self._set_hand_payload_mass()
        self._set_wrist_payload_mass()
        
    def cal_equivalent_torque(self, robot_dof_positions, robot_dof_velocities, robot_dof_accelerations):
        # shape: (num_envs, num_dofs), type: ndarray
        q = robot_dof_positions.clone().detach().cpu().numpy()[:, self.joint_usd_to_joint_urdf]
        v = robot_dof_velocities.clone().detach().cpu().numpy()[:, self.joint_usd_to_joint_urdf]
        a = robot_dof_accelerations.clone().detach().cpu().numpy()[:, self.joint_usd_to_joint_urdf]

        # Calculate equivalent torque
        tau = np.zeros_like(q)
        # Loop over num_envs
        for i in range(q.shape[0]):
            tau[i] = pin.rnea(self.pin_model, self.pin_data, q[i], v[i], a[i])   # type:ignore
        
        # Equivalent external torque
        # tau_ext = tau - self.inertias * a

        # shape: (num_envs, num_dofs), type: tensor
        return torch.tensor(tau[:, self.joint_urdf_to_joint_usd], device=self.device)  # Return corresponding joint torques
    
    def _get_observations(self) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        output = {
            "model": self.compute_model_observation(),
            "operator": self.compute_operator_observation(),
        }
        
        return {
            "model": self.compute_model_observation(),
            "operator": self.compute_operator_observation(),
            "policy": output,
        }

    def _get_rewards(self) -> torch.Tensor:
        return - ((36 / (2 * torch.pi)) ** 2) * self._reward_tracking()# - 0.1 * self._reward_delta_smoothness()
    
    def _set_wrist_payload_mass(self):
        set_masses(self.payload1, self.wrist_payload_mass, self.payload1._ALL_INDICES) # type: ignore
        set_masses(self.payload2, self.wrist_payload_mass, self.payload2._ALL_INDICES) # type: ignore

    def _set_hand_payload_mass(self):
        set_masses(self.payload3, self.hand_payload_mass[:, 0:1], self.payload3._ALL_INDICES) # type: ignore
        set_masses(self.payload4, self.hand_payload_mass[:, 1:2], self.payload4._ALL_INDICES) # type: ignore

    def _compute_metrics(self, play_runs_by_motion, save_path: str | None = None, combined_csv_path: str | None = None):
        # play_runs_by_motion: Dict[str, List[List[Dict]]], where each inner List[Dict] is one run sequence
        mass_levels = np.unique(self._motion_loader.payload_sequence.cpu().numpy()).tolist()
        mass_levels.sort()
        single_joint = self._motion_loader.joint_index is not None

        # bins to aggregate per-run metrics
        def init_bins():
            return {m: [] for m in mass_levels}
        policy_bins = init_bins()
        sim_bins = init_bins() if self._motion_loader.sim_dof_positions is not None else None
        eff_bins = init_bins()
        eff_bins_sim = init_bins() if self._motion_loader.sim_dof_positions is not None else None
        
        # New gap metrics bins
        large_gap_ratio_bins = init_bins()
        gap_iqr_bins = init_bins()
        gap_range_bins = init_bins()
        
        # Upper body joint area metric bins
        upper_body_area_bins = init_bins()
        
        # Per-joint upper body area metric bins
        per_joint_upper_body_area_bins = init_bins()

        # helper to compute per-run MPJAE over a run sequence
        def run_mpjae(seq, key):
            errors = np.abs(np.stack([step[key] for step in seq])[:, 0])
            return float(np.mean(errors))

        # helper to compute per-run EEF error norm (L2 of xyz error aggregated both hands)
        def run_eff_err_norm(seq, prefix):
            l = np.stack([step[f'{prefix}left_hand_pos'] for step in seq])[:, 0]
            r = np.stack([step[f'{prefix}right_hand_pos'] for step in seq])[:, 0]
            tl = np.stack([step['target_left_hand_pos'] for step in seq])[:, 0]
            tr = np.stack([step['target_right_hand_pos'] for step in seq])[:, 0]
            err = np.concatenate([l - tl, r - tr], axis=1)
            return float(np.linalg.norm(np.mean(np.abs(err), axis=0)))

        # helper to compute large gap ratio (ratio of gaps >= 0.5 rad)
        def run_large_gap_ratio(seq, key):
            errors = np.abs(np.stack([step[key] for step in seq]))
            large_gap_count = np.sum(errors >= 0.5)
            total_points = len(errors.flatten())
            return float(large_gap_count / total_points) if total_points > 0 else 0.0

        # helper to compute gap IQR (Interquartile Range)
        def run_gap_iqr(seq, key):
            errors = np.abs(np.stack([step[key] for step in seq]))
            q75 = np.percentile(errors, 75)
            q25 = np.percentile(errors, 25)
            return float(q75 - q25)

        # helper to compute gap range (max - min)
        def run_gap_range(seq, key):
            errors = np.abs(np.stack([step[key] for step in seq]))
            return float(np.max(errors) - np.min(errors))

        # helper to compute upper body joint area (sim-real difference area)
        def run_upper_body_area(seq, key):
            if 'sim_joint_pos_diff' not in seq[-1]:
                return 0.0  # No sim data available
            
            # Get sim-real differences for upper body joints
            sim_real_diffs = np.stack([step['sim_joint_pos_diff'] for step in seq])
            
            # Define upper body joint indices (excluding wrist joints)
            # Based on joint_sequence_wo_wrist: torso, shoulders, elbows
            upper_body_indices = list(range(sim_real_diffs.shape[2]))  # All joints are upper body joints
            
            # Calculate area for each upper body joint
            total_area = 0.0
            for joint_idx in upper_body_indices:
                if joint_idx < sim_real_diffs.shape[2]:  # Fix: use shape[2] for (timesteps, 1, joints)
                    joint_errors = np.abs(sim_real_diffs[:, 0, joint_idx])  # Fix: use [:, 0, joint_idx]
                    # Calculate area using trapezoidal rule
                    area = np.trapz(joint_errors)
                    total_area += area
            
            return float(total_area)

        # helper to compute per-joint upper body area (sim-real difference area)
        def run_per_joint_upper_body_area(seq, key):
            if 'sim_joint_pos_diff' not in seq[-1]:
                return np.zeros(10)  # Return zeros for all joints if no sim data
            
            # Get sim-real differences for upper body joints
            sim_real_diffs = np.stack([step['sim_joint_pos_diff'] for step in seq])
            upper_body_indices = list(range(sim_real_diffs.shape[2]))  # All joints are upper body joints
            
            per_joint_areas = []
            for joint_idx in range(sim_real_diffs.shape[2]):
                if joint_idx in upper_body_indices:
                    joint_errors = np.abs(sim_real_diffs[:, 0, joint_idx])
                    area = np.trapz(joint_errors)
                    per_joint_areas.append(float(area))
                else:
                    per_joint_areas.append(0.0)  # Non-upper body joints
            return np.array(per_joint_areas)

        # aggregate
        for m in range(self._motion_loader.motion_num):
            runs = play_runs_by_motion.get(str(m), [])
            for seq in runs:
                mlevel = seq[0]['payload_mass']
                # policy joint error
                policy_bins[mlevel].append(run_mpjae(seq, 'joint_pos_diff'))
                # sim joint error
                if sim_bins is not None and 'sim_joint_pos' in seq[-1]:
                    sim_bins[mlevel].append(run_mpjae(seq, 'sim_joint_pos_diff'))
                # EEF
                eff_bins[mlevel].append(run_eff_err_norm(seq, ''))
                if eff_bins_sim is not None and 'sim_left_hand_pos' in seq[-1]:
                    eff_bins_sim[mlevel].append(run_eff_err_norm(seq, 'sim_'))
                
                # New gap metrics (using joint_pos_diff to evaluate policy performance)
                # Note: These metrics measure policy error (robot_joint_pos - real_joint_pos)
                # If policy is well-trained, errors may decrease with larger payloads
                # If you want to measure sim-real gap instead, use 'sim_joint_pos_diff'
                large_gap_ratio_bins[mlevel].append(run_large_gap_ratio(seq, 'joint_pos_diff'))
                gap_iqr_bins[mlevel].append(run_gap_iqr(seq, 'joint_pos_diff'))
                gap_range_bins[mlevel].append(run_gap_range(seq, 'joint_pos_diff'))
                
                # Upper body joint area metric
                upper_body_area_bins[mlevel].append(run_upper_body_area(seq, 'sim_joint_pos_diff'))
                
                # Per-joint upper body area metric
                per_joint_upper_body_area_bins[mlevel].append(run_per_joint_upper_body_area(seq, 'sim_joint_pos_diff'))

        # prepare combined CSV if requested
        combined_lines = []

        # pretty print table
        def print_table(title, bins):
            print(title)
            header = ["Mass", "Mean ± Std", "N"]
            print(f"{header[0]:>8} | {header[1]:>20} | {header[2]:>3}")
            print("-" * 38)
            for m in mass_levels:
                vals = bins[m]
                n = len(vals)
                if n == 0:
                    mean, std = 0.0, 0.0
                else:
                    mean = float(np.mean(vals))
                    std = float(np.std(vals))
                print(f"{m:8.3f} | {mean:9.4f} ± {std:9.4f} | {n:3d}")
            print()

        # pretty print table for per-joint data
        def print_per_joint_table(title, bins, joint_names=None):
            print(title)
            if joint_names is None:
                joint_names = [f"Joint_{i}" for i in range(10)]  # Default to 10 joints
            
            # Print header
            header = ["Mass"] + joint_names
            print(" | ".join([f"{h:>8}" for h in header]))
            print("-" * (8 * (len(header) + 1)))
            
            for m in mass_levels:
                vals = bins[m]
                if len(vals) == 0:
                    row = [f"{m:8.3f}"] + ["0.0000"] * len(joint_names)
                else:
                    # Calculate mean across runs for each joint
                    joint_means = np.mean(vals, axis=0)
                    row = [f"{m:8.3f}"] + [f"{mean:8.4f}" for mean in joint_means]
                print(" | ".join(row))
            print()

        # append a section into the combined CSV with first six single-run results and aggregates
        def append_section_csv(section_title, bins):
            if combined_csv_path is None:
                return
            headers = ["Mass"] + [f"run_{i+1}" for i in range(6)] + ["mean", "std", "N"]
            combined_lines.append(f"# {section_title}")
            combined_lines.append(",".join(headers))
            for m in mass_levels:
                vals = bins[m]
                first6 = vals[:6]
                if len(first6) < 6:
                    first6 = first6 + [float('nan')] * (6 - len(first6))
                n = len(vals)
                mean = float(np.mean(vals)) if n > 0 else 0.0
                std = float(np.std(vals)) if n > 0 else 0.0
                row = [f"{m:.3f}"] + [f"{v:.4f}" if not np.isnan(v) else "" for v in first6] + [f"{mean:.4f}", f"{std:.4f}", f"{n}"]
                combined_lines.append(",".join(row))
            combined_lines.append("")

        # append per-joint data into the combined CSV
        def append_per_joint_csv(section_title, bins, joint_names=None):
            if combined_csv_path is None:
                return
            
            # Determine joint_names length dynamically from actual data
            if joint_names is None:
                # Find the first non-empty bin to determine the actual number of joints
                actual_joint_count = None
                for m in mass_levels:
                    vals = bins[m]
                    if len(vals) > 0:
                        actual_joint_count = len(vals[0])
                        break
                
                if actual_joint_count is None:
                    actual_joint_count = 10  # fallback to default
                
                joint_names = [f"Joint_{i}" for i in range(actual_joint_count)]
            
            # Create headers for per-joint data
            headers = ["Mass"] + [f"run_{i+1}_{joint}" for i in range(6) for joint in joint_names] + [f"mean_{joint}" for joint in joint_names] + [f"std_{joint}" for joint in joint_names] + ["N"]
            combined_lines.append(f"# {section_title}")
            combined_lines.append(",".join(headers))
            
            for m in mass_levels:
                vals = bins[m]
                if len(vals) == 0:
                    # No data case
                    row = [f"{m:.3f}"] + [""] * (6 * len(joint_names)) + ["0.0000"] * (2 * len(joint_names)) + ["0"]
                else:
                    # Calculate statistics for each joint
                    joint_means = np.mean(vals, axis=0)
                    joint_stds = np.std(vals, axis=0)
                    
                    # First 6 runs for each joint
                    first6_runs = vals[:6]
                    if len(first6_runs) < 6:
                        first6_runs = first6_runs + [np.zeros_like(vals[0])] * (6 - len(first6_runs))
                    
                    row = [f"{m:.3f}"]
                    # Add first 6 runs data for each joint
                    for run_idx in range(6):
                        for joint_idx in range(len(joint_names)):
                            if run_idx < len(vals) and joint_idx < len(first6_runs[run_idx]):
                                row.append(f"{first6_runs[run_idx][joint_idx]:.4f}")
                            else:
                                row.append("")
                    # Add means and stds for each joint
                    for joint_idx in range(len(joint_names)):
                        if joint_idx < len(joint_means):
                            row.append(f"{joint_means[joint_idx]:.4f}")
                        else:
                            row.append("0.0000")
                    for joint_idx in range(len(joint_names)):
                        if joint_idx < len(joint_stds):
                            row.append(f"{joint_stds[joint_idx]:.4f}")
                        else:
                            row.append("0.0000")
                
                combined_lines.append(",".join(row))
            combined_lines.append("")

        print("######################################################################")
        # New gap metrics tables and CSV sections
        print_table("Large Gap Ratio (>=0.5 rad) by Mass", large_gap_ratio_bins)
        append_section_csv("Large Gap Ratio (>=0.5 rad) by Mass", large_gap_ratio_bins)
        print_table("Gap IQR (rad) by Mass", gap_iqr_bins)
        append_section_csv("Gap IQR (rad) by Mass", gap_iqr_bins)
        print_table("Gap Range (rad) by Mass", gap_range_bins)
        append_section_csv("Gap Range (rad) by Mass", gap_range_bins)
        print("######################################################################")

        # write combined CSV once
        if combined_csv_path is not None and len(combined_lines) > 0:
            with open(combined_csv_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(combined_lines))
    def close(self):
        super().close()
        if self.mode == "play":
            print("save figures")
            # plot and save
            from .utils.plot_play import plot_joint_positions, plot_joint_velocity, plot_joint_torque, plot_joint_acc

            script_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(script_dir, f'logs/plays/{timestamp}')
            os.makedirs(save_path, exist_ok=True)

            frequency = 50
            motion_num = 1

            print(self.play_data_history.keys())

            # validate we have at least min_runs per motion
            for i in range(self._motion_loader.motion_num):
                if len(self.play_data_history.get(str(i), [])) < self.min_runs:
                    raise ValueError(f"Motion {i} has only {len(self.play_data_history.get(str(i), []))} runs, expected at least {self.min_runs}")

            # save raw sim pose if recording
            if self.cfg.record_sim_mode:
                all_sim_joint_pos = []
                for m in range(self._motion_loader.motion_num):
                    for seq in self.play_data_history[str(m)]:
                        all_sim_joint_pos.append(np.stack([r["delta_joint_pos"] for r in seq])[:, 0])
                np.savez(os.path.join(save_path, "all_sim_joint_pos.npz"), all_sim_joint_pos=np.array(all_sim_joint_pos, dtype=object))

            # compute and print metrics (mean ± std tables) and save a combined CSV
            task_name = getattr(self.cfg, 'task_name', 'humanoid_operator')
            timestamp_tag = os.path.basename(save_path)
            combined_csv_path = os.path.join(save_path, f"{timestamp_tag}_{task_name}.csv")
            self._compute_metrics(self.play_data_history, save_path, combined_csv_path)

            # plots: plot the first run of each motion for brevity
            episode_datas = []
            for i in range(self._motion_loader.motion_num):
                if str(i) not in self.play_data_history:
                    raise ValueError(f"Motion {i} not found in play_data_history")
                raw = self.play_data_history[str(i)][0]
                ep_path = os.path.join(save_path, f'episode_{i}')
                os.makedirs(ep_path, exist_ok=True)

                delta_joint_pos = np.stack([r["delta_joint_pos"] for r in raw])
                real_joint_pos = np.stack([r["real_joint_pos"] for r in raw])
                joint_target_pos = np.stack([r["joint_target_pos"] for r in raw])

                if self._motion_loader.sim_dof_positions is not None and 'sim_joint_pos' in raw[-1]:
                    sim_joint_pos = np.stack([r["sim_joint_pos"] for r in raw])
                    data = np.stack([delta_joint_pos, sim_joint_pos, joint_target_pos, real_joint_pos], axis=2)
                else:
                    # Create 4D data format compatible with plot_joint_positions
                    # Format: (time_length, 1, 4, 10) - delta_joint_pos, sim_joint_pos (placeholder), joint_target_pos, real_joint_pos
                    sim_joint_pos = np.zeros_like(delta_joint_pos)  # Placeholder for sim data
                    data = np.stack([delta_joint_pos, sim_joint_pos, joint_target_pos, real_joint_pos], axis=2)
                episode_datas.append(data)
                plot_joint_positions(data, ep_path, frequency, motion_num, self._motion_loader.joint_sequence)
                # plot_joint_velocity(data, save_path, frequency, motion_num, self._motion_loader.joint_sequence)
                # plot_joint_torque(data, save_path, frequency, motion_num, self._motion_loader.joint_sequence)
                # plot_joint_acc(data, save_path, frequency, motion_num, self._motion_loader.joint_sequence)
            np.savez(os.path.join(save_path, "all_episode_datas.npz"), task_data=np.array(episode_datas, dtype=object),
                     payloads=self._motion_loader.payload_sequence.cpu().numpy())

    #########################################################
    # Functions for Operator
    #########################################################
    def _init_sensor_positions(self):
        self.sensor_positions = get_sensor_positions(self.robot.data.joint_names, self.cfg.sensors_positions).to(self.device)
        self.sensor_positions = self.sensor_positions.repeat(self.num_sub_environments, 1)

    def _sample_sub_environments(self, min_available_length: int = 1):
        motion_indices, time_indices = self._motion_loader.sample_indices(self.num_sub_environments, randomize_start=True, min_available_length=min_available_length)
        self.motion_indices[:] = motion_indices.repeat_interleave(self.num_sensor_positions, dim=0)
        self.time_indices[:] = time_indices.repeat_interleave(self.num_sensor_positions, dim=0)
        self.wrist_payload_mass[:] = self._motion_loader.payload_sequence[self.motion_indices] + 0.001
        self._set_wrist_payload_mass()

    def sample_all_environments(self, env_ids: torch.Tensor | None = None, min_available_length: int = 1):
        if env_ids is None:
            env_ids = self._ALL_INDICES
        if len(env_ids) == 0:
            return
        
        motion_indices, time_indices = self._motion_loader.sample_indices(len(env_ids), randomize_start=True, min_available_length=min_available_length) # type: ignore
        self.motion_indices[env_ids] = motion_indices
        self.time_indices[env_ids] = time_indices

        joint_pos = self._motion_loader.dof_positions[self.motion_indices, self.time_indices]
        joint_vel = self._motion_loader.dof_velocities[self.motion_indices, self.time_indices]
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids) # type: ignore

        if self._motion_loader.hand_marker is None:
            self.hand_payload_mass[env_ids] = 0.001
            self.wrist_payload_mass[env_ids] = self._motion_loader.payload_sequence[self.motion_indices[env_ids]] + 0.001
        else:
            self.hand_payload_mass[env_ids] = 0.001
            self.hand_payload_mass[env_ids, self._motion_loader.hand_marker[self.motion_indices[env_ids]]] = self._motion_loader.payload_sequence[self.motion_indices[env_ids]] + 0.001
            self.wrist_payload_mass[env_ids] = 0.001

        self._set_wrist_payload_mass()
        self._set_hand_payload_mass()

        self.last_delta_action[env_ids] *= 0
        self.model_history[env_ids] *= 0

    def _raw_step_simulator(self):
        # NOTE this function only works for implicit actuators

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # self.before_positions = self.robot.data.joint_pos.clone()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
        
        # self.after_positions = self.robot.data.joint_pos
        # self.step_velocity[:] = (self.after_positions - self.before_positions) / self.step_dt

    def _pre_set_sensor_data(self):
        joint_pos = self.robot.data.joint_pos[:, self._motion_loader.joint_sequence_index]
        joint_vel = self.robot.data.joint_vel[:, self._motion_loader.joint_sequence_index]
        self.pre_sub_env_joint_states = torch.cat([
                joint_pos,
                joint_vel * self.step_dt,
        ], dim=1).view(self.num_sub_environments, self.num_sensor_positions, self.cfg.sensor_dim)

    def _set_sensor_data(self):
        joint_pos = self.robot.data.joint_pos[:, self._motion_loader.joint_sequence_index]
        joint_vel = self.robot.data.joint_vel[:, self._motion_loader.joint_sequence_index]

        self.sub_env_sensor_data[:] = torch.cat([
                joint_pos,
                joint_vel * self.step_dt,
        ], dim=1).view(self.num_sub_environments, self.num_sensor_positions, self.cfg.sensor_dim)
        if self.cfg.delta_sensor_value:
            self.sub_env_sensor_data[:] = self.sub_env_sensor_data - self.pre_sub_env_joint_states

    def set_sensor_data(self, sensor_data: torch.Tensor):
        self.sensor_data[:] = sensor_data
        
    def compute_operator_observation(self) -> Dict[str, torch.Tensor]:
        current_action = self._motion_loader.dof_target_pos[self.motion_indices, self.time_indices][:, self._motion_loader.joint_sequence_index]
        real_joint_pos = self._motion_loader.dof_positions[self.motion_indices, self.time_indices][:, self._motion_loader.joint_sequence_index]
        real_joint_vel = self._motion_loader.dof_velocities[self.motion_indices, self.time_indices][:, self._motion_loader.joint_sequence_index]
        payload = self._motion_loader.payload_sequence[self.motion_indices]
        
        branch_obs = torch.cat([
            self.sensor_data.flatten(1, 2),
        ], dim=1)
        trunk_obs = torch.cat([
            current_action,
            # payload,
            # self.robot.data.joint_pos[:, self._motion_loader.joint_sequence_index],
        ], dim=1)
        critic_obs = torch.cat([
            self.sensor_data.flatten(1, 2),
            current_action,
            self.robot.data.joint_pos[:, self._motion_loader.joint_sequence_index],
            self.robot.data.joint_vel[:, self._motion_loader.joint_sequence_index],
            self.robot.data.joint_acc[:, self._motion_loader.joint_sequence_index],
            real_joint_pos,
            real_joint_vel,
            self.wrist_payload_mass,
            self.hand_payload_mass,
            self.robot_mass,
        ], dim=1)

        if self.add_noise:
            branch_obs = branch_obs + torch.rand_like(branch_obs) * 0.01
            trunk_obs = trunk_obs + torch.rand_like(trunk_obs) * 0.01

        return {
            "branch": branch_obs.clone(),
            "trunk": trunk_obs.clone(),
            "critic": critic_obs.clone(),
        }
    
    def step_operator(self, delta_action: torch.Tensor,
                       motion_coords: Tuple[torch.Tensor, torch.Tensor] | None = None):
        ''' This function is used to step the operator, this could be called multiple times '''

        if not motion_coords is None:
            self.motion_indices[:], self.time_indices[:] = motion_coords
        self.delta_action[:] = delta_action

        self.apply_action[:] = self._motion_loader.dof_target_pos[self.motion_indices, self.time_indices]
        self.apply_action[:, self._motion_loader.joint_sequence_index] += self.delta_action
        self.robot.set_joint_position_target(self.apply_action)

        self._raw_step_simulator()
        # # update time_indices
        unfinished_motion = self._motion_loader.motion_len[self.motion_indices] > (self.time_indices + 1)
        self.time_indices[unfinished_motion] += 1

        rewards = self._get_rewards()
        rewards = rewards * unfinished_motion
        dones = ~unfinished_motion

        self.sample_all_environments(env_ids=self._ALL_INDICES[dones])

        self.last_delta_action[:] = self.delta_action
        return None, rewards, dones, {'episode': {
            'joint_pos_diff': torch.abs((self._motion_loader.dof_positions[self.motion_indices, self.time_indices] - self.robot.data.joint_pos)[:, self._motion_loader.joint_sequence_index]) * (360 / 6.28),
            'joint_vel_diff': torch.abs((self._motion_loader.dof_velocities[self.motion_indices, self.time_indices] - self.robot.data.joint_vel)[:, self._motion_loader.joint_sequence_index]) * (360 / 6.28),
        }}


    def compute_model_observation(self, add_noise: bool = False) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self._motion_loader.joint_sequence_index]
        joint_vel = self.robot.data.joint_vel[:, self._motion_loader.joint_sequence_index]
        joint_target = self.robot.data.joint_pos_target[:, self._motion_loader.joint_sequence_index]
        # joint_acc = self.robot.data.joint_acc[:, self._motion_loader.joint_sequence_index]
        if not self.add_model_history:
            if self._motion_loader.hand_marker is None:
                payload_mass = self.wrist_payload_mass
            else:
                payload_mass = self.hand_payload_mass[self._ALL_INDICES, self._motion_loader.hand_marker[self.motion_indices]][:, None]
            model_obs = torch.cat([joint_pos, joint_vel, payload_mass], dim=1).clone()
        else:
            model_obs = torch.cat([joint_pos, self.model_history.flatten(1, 2)], dim=1).clone()
            self.model_history = self.model_history.roll(1, dims=1)
            self.model_history[:, 0, :] = torch.cat([joint_pos, joint_vel, joint_target], dim=1).clone()
        # import pdb; pdb.set_trace()

        if self.add_noise and add_noise:
            model_obs = model_obs + torch.rand_like(model_obs) * 0.01
        return model_obs

    def sample_all_dynamics(self, sub_env_consistent: bool = False, sample_payload: bool = True):
        default_mass = self.robot.data.default_mass
        default_mass = default_mass * (torch.rand(self.num_envs, self.robot.num_bodies, device="cpu") * (self.cfg.robot_mass_range[1] - self.cfg.robot_mass_range[0]) + self.cfg.robot_mass_range[0])
        if sub_env_consistent:
            default_mass[:] = default_mass[::self.num_sensor_positions].repeat_interleave(self.num_sensor_positions, dim=0)
        
        self.robot_mass[:] = default_mass.to(device=self.device)
        set_masses(self.robot, default_mass, self._ALL_INDICES)

        if sample_payload:
            self.wrist_payload_mass[:] = torch.rand(self.num_envs, 1, device=self.device) * self.cfg.max_payload_mass + 0.001
            if sub_env_consistent:
                self.wrist_payload_mass[:] = self.wrist_payload_mass[::self.num_sensor_positions].repeat_interleave(self.num_sensor_positions, dim=0)
            self._set_wrist_payload_mass()

            self.hand_payload_mass[:] = 0.001
            self.hand_payload_mass[self._ALL_INDICES, torch.randint(0, self.hand_payload_mass.shape[1], (self.num_envs,), device=self.device)] = torch.rand(self.num_envs, device=self.device) * self.cfg.max_payload_mass + 0.001
            if sub_env_consistent:
                self.hand_payload_mass[:] = self.hand_payload_mass[::self.num_sensor_positions].repeat_interleave(self.num_sensor_positions, dim=0)
            self._set_hand_payload_mass()

    def reset_all_dynamics(self):
        reset_masses(self.robot, self._ALL_INDICES)
    
    def compute_model_pairs(self, add_noise: bool = False) -> Dict[str, torch.Tensor]:
        joint_limit_delta = (self.joint_upper_limits - self.joint_lower_limits)[:, self._motion_loader.joint_sequence_index]
        joint_lower_limits = self.joint_lower_limits[:, self._motion_loader.joint_sequence_index]
        joint_vel_limits = self.joint_vel_limits[:, self._motion_loader.joint_sequence_index]

        sub_env_limit_delta = joint_limit_delta[::self.num_sensor_positions]
        sub_env_lower_limits = joint_lower_limits[::self.num_sensor_positions]
        sub_env_vel_limits = joint_vel_limits[::self.num_sensor_positions]
        
        joint_pos = torch.zeros(self.num_sub_environments, self.num_dofs, device=self.device)
        joint_vel = torch.zeros(self.num_sub_environments, self.num_dofs, device=self.device)

        joint_pos[:, self._motion_loader.joint_sequence_index] = 0.8 * (torch.rand(self.num_sub_environments, len(self._motion_loader.joint_sequence_index), device=self.device) * sub_env_limit_delta + sub_env_lower_limits)
        joint_vel[:, self._motion_loader.joint_sequence_index] = 0.5 * (torch.rand(self.num_sub_environments, len(self._motion_loader.joint_sequence_index), device=self.device) * sub_env_vel_limits)
        
        joint_pos = joint_pos.repeat_interleave(self.num_sensor_positions, dim=0)
        joint_vel = joint_vel.repeat_interleave(self.num_sensor_positions, dim=0)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        self.sample_all_dynamics(sub_env_consistent=True)
        if not self.add_model_history:
            if self.cfg.delta_sensor_position:
                self.robot.set_joint_position_target(self.sensor_positions + self.robot.data.joint_pos)
            else:
                self.robot.set_joint_position_target(self.sensor_positions)

            obs = self.compute_model_observation(add_noise)[::self.num_sensor_positions]
            self._pre_set_sensor_data()
            for _ in range(self.cfg.sensor_decimation):
                self._raw_step_simulator()
            self._set_sensor_data()

            return {
                "obs": obs.clone(),
                "sensor": self.sub_env_sensor_data.clone()
            }

        else:
            self.model_history[:] = 0.
            fill_length = np.random.randint(0, self.cfg.model_initial_fill_length * 10)
            fill_length = min(fill_length, self.cfg.model_initial_fill_length)

            for i in range(fill_length):
                # NOTE: We first step random points
                random_target = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
                random_target[:, self._motion_loader.joint_sequence_index] = 0.8 * (torch.rand_like(self.robot.data.joint_pos[:, self._motion_loader.joint_sequence_index]) * joint_limit_delta + joint_lower_limits)
                self.robot.set_joint_position_target(random_target)

                obs = self.compute_model_observation(add_noise)
                self._raw_step_simulator()

            obs = self.compute_model_observation(add_noise)
            obs = obs[::self.num_sensor_positions]
            joint_pos = self.robot.data.joint_pos[::self.num_sensor_positions].repeat_interleave(self.num_sensor_positions, dim=0)
            joint_vel = self.robot.data.joint_vel[::self.num_sensor_positions].repeat_interleave(self.num_sensor_positions, dim=0)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel * 0)
            if self.cfg.delta_sensor_position:
                self.robot.set_joint_position_target(self.sensor_positions + joint_pos)
            else:
                self.robot.set_joint_position_target(self.sensor_positions)

            self._pre_set_sensor_data()
            for _ in range(self.cfg.sensor_decimation):
                self._raw_step_simulator()
            self._set_sensor_data()

            return {
                "obs": obs.clone(),
                "sensor": self.sub_env_sensor_data.clone()
            }
    # ------------------------------------ reward functions ------------------------------------
    def _reward_tracking(self):
        # robot current state
        robot_dof_positions = self.robot.data.joint_pos     # shape: (num_envs, num_dofs)
        robot_dof_velocities = self.robot.data.joint_vel    # shape: (num_envs, num_dofs)

        # sampled state from MotionLoader
        real_dof_positions = self._motion_loader.dof_positions[self.motion_indices, self.time_indices]   # shape: (num_envs, num_dofs)
        real_dof_velocities = self._motion_loader.dof_velocities[self.motion_indices, self.time_indices] # shape: (num_envs, num_dofs)
        
        joint_index = self._motion_loader.joint_sequence_index   # shape: (10,)

        # calculate rewards
        position_diff = (robot_dof_positions - real_dof_positions) ** 2     # shape: (num_envs, 10)
        velocity_diff = (robot_dof_velocities - real_dof_velocities) ** 2   # shape: (num_envs, 10)
        
        position_diff = torch.mean(position_diff[:, joint_index], dim=1)   # shape: (num_envs, )
        velocity_diff = torch.mean(velocity_diff[:, joint_index], dim=1)   # shape: (num_envs, )

        return position_diff + velocity_diff * 1e-2
    
    def _reward_delta_smoothness(self):
        last_action = self.last_delta_action
        current_action = self.delta_action

        reward = (current_action - last_action) ** 2
        reward = torch.clamp(reward - 4e-1, min=0).sum(dim=1)
        return reward * (~torch.all(torch.abs(last_action) < 1e-2, dim=1)) * 0
    
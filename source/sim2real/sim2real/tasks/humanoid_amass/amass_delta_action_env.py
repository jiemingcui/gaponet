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
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate

import pinocchio as pin
from copy import deepcopy

from .amass_delta_action_env_cfg import HumanoidMotorAmassEnvCfg
from .motions import MotionLoaderMotor


class HumanoidMotorAmassEnv(DirectRLEnv):
    cfg: HumanoidMotorAmassEnvCfg

    def __init__(self, cfg: HumanoidMotorAmassEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        self.dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        self.dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        
        self.mode = self.cfg.mode

        # load motion
        self._motion_loader = MotionLoaderMotor(motion_file=self.cfg.train_motion_file if self.mode == "train" else self.cfg.test_motion_file, device=self.device, mode=self.mode, robot_name=self.cfg.robot_name)  # type: ignore
        self.num_dofs = self._motion_loader.num_dofs

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)


        self.motion_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))

        # shape: (num_envs, history_obs_len==1, obs_len==3)
        self.amp_observation_buffer = torch.zeros((self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device)

        self.apply_action = torch.zeros((self.num_envs, self.num_dofs), device=self.device)

        if self.mode == "play":
            self.obs_history = []
            self.reset_nums = 0
            self.saved_nums = 0
            self.play_data_history = {}
            self.play_data = [[] for _ in range(self.num_envs)]
            # minimum number of runs to aggregate results (default 6)
            self.min_runs = getattr(self.cfg, 'min_runs', 4)

        if self.cfg.if_torque_input:
            self.pin_model, _, _ = pin.buildModelsFromUrdf(self.cfg.urdf_model_path, package_dirs=self.cfg.package_dirs) # type: ignore
            self.pin_data = self.pin_model.createData()

            self.joint_urdf_to_joint_usd = [self.cfg.urdf_joint_name.index(name) for name in self._motion_loader.dof_names]  # Index of joint_usd in joint_urdf
            self.joint_usd_to_joint_urdf = [self._motion_loader.dof_names.index(name) for name in self.cfg.urdf_joint_name]  # Index of joint_urdf in joint_usd

            inertias = []
            for joint_name in self.cfg.urdf_joint_name:
                joint_id = self.pin_model.getJointId(joint_name)
                inertia = self.pin_model.inertias[joint_id]
                joint = self.pin_model.joints[joint_id]
                
                if joint.shortname() == "JointModelRZ":
                    axis = np.array([0, 0, 1])
                elif joint.shortname() == "JointModelRY":
                    axis = np.array([0, 1, 0])
                elif joint.shortname() == "JointModelRX":
                    axis = np.array([1, 0, 0])
                
                axis = axis / np.linalg.norm(axis)
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
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        dof_target_pos = self._motion_loader.dof_target_pos[self.motion_indices, self.time_indices]
        delta_action = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        joint_index = self._motion_loader.joint_sequence_index
        delta_action[:, joint_index] = self.actions
        
        self.apply_action = (dof_target_pos + delta_action).clone()
        self.robot.set_joint_position_target(self.apply_action)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        joint_acc = self.robot.data.joint_acc
        
        joint_index = self._motion_loader.joint_sequence_index
        
        joint_equivalent_torque = torch.zeros_like(joint_pos, device=self.device)
        if self.cfg.if_torque_input:
            joint_equivalent_torque = self.cal_equivalent_torque(joint_pos, joint_vel, joint_acc)
        
        obs = torch.cat([
            joint_pos[:, joint_index], 
            joint_vel[:, joint_index], 
            self._motion_loader.dof_positions[self.motion_indices, self.time_indices][:, joint_index], 
            joint_acc[:, joint_index],
            joint_equivalent_torque[:, joint_index],
            ], dim=-1)
        
        self.amp_observation_buffer = self.amp_observation_buffer.roll(1, dims=1)
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": self.amp_observation_buffer.clone().view(self.num_envs, -1)}

    def _get_rewards(self) -> torch.Tensor:
        position_diff = self._reward_tracking()
        reward = -5 * position_diff
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)

        self.time_indices += 1

        if self.mode == "play":
            joint_index = self._motion_loader.joint_sequence_index
            for i in range(self.num_envs):
                self.play_data[i].append({
                    "delta_joint_pos": self.robot.data.joint_pos[i:i+1, joint_index].clone().cpu().numpy(),
                    "real_joint_pos": self._motion_loader.dof_positions[self.motion_indices[i:i+1], self.time_indices[i:i+1]][:, joint_index].clone().cpu().numpy(),
                    "joint_target_pos": self._motion_loader.dof_target_pos[self.motion_indices[i:i+1], self.time_indices[i:i+1]][:, joint_index].clone().cpu().numpy(),
                    "joint_pos_diff": self.robot.data.joint_pos[i:i+1, joint_index].clone().cpu().numpy() - self._motion_loader.dof_positions[self.motion_indices[i:i+1], self.time_indices[i:i+1]][:, joint_index].clone().cpu().numpy(),
                    "payload_mass": self._motion_loader.payload_sequence[self.motion_indices[i:i+1]].clone().cpu().item(),
                })

        motion_done = self.time_indices >= self._motion_loader.motion_len[self.motion_indices] - 1
        return died, motion_done

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        env_ids: The environment IDs to reset. If None, reset all environments.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES  # type: ignore

        if self.mode == "play" and len(env_ids) > 0:
            self.reset_nums += 1
            for env_id in env_ids.tolist():
                if len(self.play_data[env_id]) > 0:
                    motion_idx = str(self.motion_indices[env_id].item())
                    if motion_idx not in self.play_data_history:
                        self.play_data_history[motion_idx] = []
                    self.play_data_history[motion_idx].append(deepcopy(self.play_data[env_id]))
                    self.play_data[env_id] = []
            # progress print
            counts = {i: len(self.play_data_history.get(str(i), [])) for i in range(self._motion_loader.motion_num)}
            print("Motion run counts:", counts)
            # stop when each motion has at least min_runs
            if all(len(self.play_data_history.get(str(i), [])) >= self.min_runs for i in range(self._motion_loader.motion_num)):
                self.close()
                exit()

        self.motion_indices[env_ids], self.time_indices[env_ids] = self._motion_loader.sample_indices(len(env_ids))   

        dof_positions = self._motion_loader.dof_positions[self.motion_indices[env_ids], self.time_indices[env_ids]]
        
        self.robot.reset(env_ids)  # type: ignore
        super()._reset_idx(env_ids) # type: ignore

        # Reset robot joint positions to target positions
        joint_pos = dof_positions.clone()
        joint_vel = torch.zeros_like(joint_pos, device=self.device)
        joint_acc = torch.zeros_like(joint_pos, device=self.device)
        joint_equivalent_torque = torch.zeros_like(joint_pos, device=self.device)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        joint_index = self._motion_loader.joint_sequence_index
        
        # Build observations
        amp_observations = torch.cat([
            joint_pos[:, joint_index], 
            joint_vel[:, joint_index], 
            dof_positions[:, joint_index], 
            joint_acc[:, joint_index],
            joint_equivalent_torque[:, joint_index],
            ], dim=-1)
        amp_observations = amp_observations.unsqueeze(1).repeat(1, self.cfg.num_amp_observations, 1)  

        self.amp_observation_buffer[env_ids] = amp_observations

    def _compute_metrics(self, play_runs_by_motion, save_path: str | None = None, combined_csv_path: str | None = None):
        # play_runs_by_motion: Dict[str, List[List[Dict]]], where each inner List[Dict] is one run sequence
        # Dynamically determine mass levels from actual data
        mass_levels = set()
        for motion_runs in play_runs_by_motion.values():
            for seq in motion_runs:
                if len(seq) > 0 and 'payload_mass' in seq[0]:
                    mass_levels.add(seq[0]['payload_mass'])
        mass_levels = sorted(list(mass_levels))
        print(f"Detected mass levels: {mass_levels}")

        # bins to aggregate per-run metrics
        def init_bins():
            return {m: [] for m in mass_levels}
        policy_bins = init_bins()
        
        # New gap metrics bins
        large_gap_ratio_bins = init_bins()
        gap_iqr_bins = init_bins()
        gap_range_bins = init_bins()
        
        # Upper body joint area metric bins
        upper_body_area_bins = init_bins()
        
        # Per-joint metrics bins (only for Upper Body Joint Area)
        per_joint_upper_body_area_bins = init_bins()

        # helper to compute per-run MPJAE over a run sequence
        def run_mpjae(seq, key):
            errors = np.abs(np.stack([step[key] for step in seq])[:, 0])
            return float(np.mean(errors))

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

        # helper to compute upper body joint area (policy-real difference area)
        def run_upper_body_area(seq, key):
            # Get policy-real differences for upper body joints
            policy_real_diffs = np.stack([step[key] for step in seq])
            
            # Define upper body joint indices (excluding wrist joints)
            # Based on joint_sequence_wo_wrist: torso, shoulders, elbows
            # For AMASS, we use policy-real difference instead of sim-real
            upper_body_indices = list(range(policy_real_diffs.shape[2]))  # All joints are upper body joints
            
            # Calculate area for each upper body joint
            total_area = 0.0
            for joint_idx in upper_body_indices:
                if joint_idx < policy_real_diffs.shape[2]:  # Fix: use shape[2] for (timesteps, 1, joints)
                    joint_errors = np.abs(policy_real_diffs[:, 0, joint_idx])  # Fix: use [:, 0, joint_idx]
                    # Calculate area using trapezoidal rule
                    area = np.trapz(joint_errors)
                    total_area += area
            
            return float(total_area)

        # helper to compute per-joint upper body area (policy-real difference area)
        def run_per_joint_upper_body_area(seq, key):
            # Get policy-real differences for upper body joints
            policy_real_diffs = np.stack([step[key] for step in seq])
            # All joints in this dataset are upper body joints
            num_joints = policy_real_diffs.shape[2]
            
            per_joint_areas = []
            for joint_idx in range(num_joints):
                joint_errors = np.abs(policy_real_diffs[:, 0, joint_idx])
                area = np.trapz(joint_errors)
                per_joint_areas.append(float(area))
            return np.array(per_joint_areas)

        # aggregate
        for m in range(self._motion_loader.motion_num):
            runs = play_runs_by_motion.get(str(m), [])
            for seq in runs:
                mlevel = seq[0]['payload_mass']  # Use actual payload mass from data
                # policy joint error
                policy_bins[mlevel].append(run_mpjae(seq, 'joint_pos_diff'))
                
                # New gap metrics
                large_gap_ratio_bins[mlevel].append(run_large_gap_ratio(seq, 'joint_pos_diff'))
                gap_iqr_bins[mlevel].append(run_gap_iqr(seq, 'joint_pos_diff'))
                gap_range_bins[mlevel].append(run_gap_range(seq, 'joint_pos_diff'))
                
                # Upper body joint area metric
                upper_body_area_bins[mlevel].append(run_upper_body_area(seq, 'joint_pos_diff'))
                
                # Per-joint metrics (only Upper Body Joint Area)
                per_joint_upper_body_area_bins[mlevel].append(run_per_joint_upper_body_area(seq, 'joint_pos_diff'))

        # prepare combined CSV if requested
        combined_lines = []

        # pretty print table
        def print_table(title, bins):
            print(title)
            header = ["Motion", "Mean ± Std", "N"]
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
            header = ["Motion"] + joint_names
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
            headers = ["Motion"] + [f"run_{i+1}" for i in range(6)] + ["mean", "std", "N"]
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
            if joint_names is None:
                joint_names = [f"Joint_{i}" for i in range(10)]  # Default to 10 joints
            
            # Create headers for per-joint data
            headers = ["Motion"] + [f"run_{i+1}_{joint}" for i in range(6) for joint in joint_names] + [f"mean_{joint}" for joint in joint_names] + [f"std_{joint}" for joint in joint_names] + ["N"]
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
                            if run_idx < len(vals):
                                row.append(f"{first6_runs[run_idx][joint_idx]:.4f}")
                            else:
                                row.append("")
                    # Add means and stds for each joint
                    for joint_idx in range(len(joint_names)):
                        row.append(f"{joint_means[joint_idx]:.4f}")
                    for joint_idx in range(len(joint_names)):
                        row.append(f"{joint_stds[joint_idx]:.4f}")
                    row.append(f"{len(vals)}")
                
                combined_lines.append(",".join(row))
            combined_lines.append("")

        print("######################################################################")
        print_table("Policy MPJAE (deg) by Motion", policy_bins)
        append_section_csv("Policy MPJAE (deg) by Motion", policy_bins)
        
        # New gap metrics tables and CSV sections
        print_table("Large Gap Ratio (>=0.5 rad) by Motion", large_gap_ratio_bins)
        append_section_csv("Large Gap Ratio (>=0.5 rad) by Motion", large_gap_ratio_bins)
        print_table("Gap IQR (rad) by Motion", gap_iqr_bins)
        append_section_csv("Gap IQR (rad) by Motion", gap_iqr_bins)
        print_table("Gap Range (rad) by Motion", gap_range_bins)
        append_section_csv("Gap Range (rad) by Motion", gap_range_bins)
        print_table("Upper Body Joint Area (rad·s) by Motion", upper_body_area_bins)
        append_section_csv("Upper Body Joint Area (rad·s) by Motion", upper_body_area_bins)
        
        # Per-joint metrics output (only Upper Body Joint Area)
        print_per_joint_table("Per-Joint Upper Body Area (rad·s) by Motion", per_joint_upper_body_area_bins)
        append_per_joint_csv("Per-Joint Upper Body Area (rad·s) by Motion", per_joint_upper_body_area_bins)
        print("######################################################################")

        # write combined CSV once
        if combined_csv_path is not None and len(combined_lines) > 0:
            with open(combined_csv_path, 'w') as f:
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

            # compute and print metrics (mean ± std tables) and save a combined CSV
            task_name = getattr(self.cfg, 'task_name', 'humanoid_amass')
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

                # Create data format compatible with plot_joint_positions
                # Format: (time_length, 1, 4, 10) - delta_joint_pos, sim_joint_pos (placeholder), joint_target_pos, real_joint_pos
                sim_joint_pos = np.zeros_like(delta_joint_pos)  # Placeholder for sim data
                data = np.stack([delta_joint_pos, sim_joint_pos, joint_target_pos, real_joint_pos], axis=2)
                episode_datas.append(data)
                plot_joint_positions(data, ep_path, frequency, motion_num, self._motion_loader.joint_sequence)
            
            np.savez(os.path.join(save_path, "all_episode_datas.npz"), task_data=np.array(episode_datas, dtype=object))

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

    # ------------------------------------ reward functions ------------------------------------
    def _reward_tracking(self):
        # Robot current state
        robot_dof_positions = self.robot.data.joint_pos     # shape: (num_envs, num_dofs)
        robot_dof_velocities = self.robot.data.joint_vel    # shape: (num_envs, num_dofs)

        # Sampled state from MotionLoader
        real_dof_positions = self._motion_loader.dof_positions[self.motion_indices, self.time_indices]   # shape: (num_envs, num_dofs)
        real_dof_velocities = self._motion_loader.dof_velocities[self.motion_indices, self.time_indices] # shape: (num_envs, num_dofs)
        
        joint_index = self._motion_loader.joint_sequence_index   # shape: (10,)

        # Calculate reward
        position_diff = (robot_dof_positions - real_dof_positions)[:, joint_index] ** 2     # shape: (num_envs, 10)
        velocity_diff = (robot_dof_velocities - real_dof_velocities)[:, joint_index] ** 2   # shape: (num_envs, 10)
        
        position_diff = torch.mean(position_diff, dim=1)   # shape: (num_envs, )
        velocity_diff = torch.mean(velocity_diff, dim=1)   # shape: (num_envs, )

        return position_diff
    
    def _reward_smoothness(self):
        # Extract joint_positions and joint_velocities
        joint_positions = self.amp_observation_buffer[:, :, 0:10]      # shape: (num_envs, history_obs_len, 10)
        joint_velocities = self.amp_observation_buffer[:, :, 10:20]    # shape: (num_envs, history_obs_len, 10)

        # Calculate smoothness of joint_positions
        # Difference: compute change between adjacent time steps
        pos_diff = joint_positions[:, 1:] - joint_positions[:, :-1]        # shape: (num_envs, history_obs_len - 1, 10)
        # Mean of squared differences as smoothness measure
        pos_smoothness = torch.mean(pos_diff**2, dim=1)       # shape: (num_envs, 10)

        # Calculate smoothness of joint_velocities
        vel_diff = joint_velocities[:, 1:] - joint_velocities[:, :-1]      # shape: (num_envs, history_obs_len - 1, 10)
        vel_smoothness = torch.mean(vel_diff**2, dim=1)       # shape: (num_envs, 10)

        pos_smoothness = torch.mean(pos_smoothness, dim=1)   # shape: (num_envs, )
        vel_smoothness = torch.mean(vel_smoothness, dim=1)   # shape: (num_envs, )
        
        # Return result with shape (num_envs, 1)
        return pos_smoothness, vel_smoothness

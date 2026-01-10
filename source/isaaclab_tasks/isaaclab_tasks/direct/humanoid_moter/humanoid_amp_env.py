# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate

from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg
from .motions import MotionLoaderMotor


class HumanoidAmpEnv(DirectRLEnv):
    cfg: HumanoidAmpEnvCfg

    def __init__(self, cfg: HumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]

        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        
        # load motion
        self._motion_loader = MotionLoaderMotor(motion_file=self.cfg.motion_file, device=self.device)
        self.num_dofs = self._motion_loader.num_dofs

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)

        # 初始化环境的 motion 和 time 索引
        self.motion_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))

        # num_envs, history_obs_len==1, obs_len==3
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

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
        delta_action = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        joint_index = self._motion_loader.joint_sequence_index[self.motion_indices]

        # delta_action扩展到（4096, 27）
        delta_action[torch.arange(self.num_envs), joint_index] = (self.action_offset + self.action_scale * self.actions)[torch.arange(self.num_envs), joint_index]

        # 根据当前 time_indices 和 motion_indices 获取目标位置
        dof_target_pos = self._motion_loader.dof_target_pos[self.motion_indices, self.time_indices]

        self.robot.set_joint_position_target(dof_target_pos + delta_action)
        

    def _get_observations(self) -> dict:
        # 当前机器人状态, s_t
        robot_dof_positions = self.robot.data.joint_pos
        robot_dof_velocities = self.robot.data.joint_vel

        # 从 MotionLoader 中采样的状态, a_t
        dof_target_pos = self._motion_loader.dof_target_pos[self.motion_indices, self.time_indices].clone()


        joint_index = self._motion_loader.joint_sequence_index[self.motion_indices]

        obs = torch.cat([
            robot_dof_positions[torch.arange(self.num_envs), joint_index].unsqueeze(1),  
            robot_dof_velocities[torch.arange(self.num_envs), joint_index].unsqueeze(1), 
            dof_target_pos[torch.arange(self.num_envs), joint_index].unsqueeze(1)
            ], dim=-1)


        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 机器人当前状态
        robot_dof_positions = self.robot.data.joint_pos
        robot_dof_velocities = self.robot.data.joint_vel

        # 从 MotionLoader 中采样的状态
        dof_positions = self._motion_loader.dof_positions[self.motion_indices, self.time_indices]
        dof_velocities = self._motion_loader.dof_velocities[self.motion_indices, self.time_indices]
        
        joint_index = self._motion_loader.joint_sequence_index[self.motion_indices]

        # 计算奖励
        position_diff = (robot_dof_positions - dof_positions)[torch.arange(self.num_envs), joint_index] ** 2
        velocity_diff = (robot_dof_velocities - dof_velocities)[torch.arange(self.num_envs), joint_index] ** 2
        reward = -position_diff #- velocity_diff
        # import pdb; pdb.set_trace()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)

        # 更新 time_indices
        self.time_indices += 1

        # 检查是否需要重置（time_indices 超过 motion_len）
        motion_done = self.time_indices >= self._motion_loader.motion_len - 1
        
        return died, time_out | motion_done

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # 随机重置 motion_indices 和 time_indices
        self.motion_indices[env_ids], self.time_indices[env_ids] = self._motion_loader.sample_indices(len(env_ids))

        # 获取对应的目标状态
        dof_target_pos = self._motion_loader.dof_target_pos[self.motion_indices[env_ids], self.time_indices[env_ids]]
        
        
        # 重置机器人状态
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 将机器人关节位置重置为目标位置
        joint_pos = dof_target_pos.clone()
        joint_vel = torch.zeros_like(joint_pos, device=self.device)  # 重置速度为 0
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        joint_index = self._motion_loader.joint_sequence_index[self.motion_indices[env_ids]]

        amp_observations = torch.cat([joint_pos[torch.arange(len(env_ids)), joint_index].unsqueeze(1), 
                                      joint_vel[torch.arange(len(env_ids)), joint_index].unsqueeze(1), 
                                      dof_target_pos[torch.arange(len(env_ids)), joint_index].unsqueeze(1)], dim=-1)

        self.amp_observation_buffer[env_ids] = amp_observations.view(len(env_ids), self.cfg.num_amp_observations, -1)




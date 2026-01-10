# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
from typing import Optional

from .joint_names import ROBOT_BODY_JOINT_NAME_DICT


class MotionLoaderMotor:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device | str, mode: str, robot_name: str) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file, allow_pickle=True)
        print(f"Loading motion data from {motion_file}...")

        self.device = device

        self._body_names = ROBOT_BODY_JOINT_NAME_DICT[f"{robot_name}_links"]
        self._dof_names = ROBOT_BODY_JOINT_NAME_DICT[f"{robot_name}_joints"]


        self.motion_index = 10       # train dataset
        self.mode = mode
        if self.mode == "play":
            self.sample_time = 0
            self.motion_index = 0    # 0-10 test dataset


        # type: List[ndarray]
        self.dof_positions_list = data["real_dof_positions"][self.motion_index:]
        self.dof_velocities_list = data["real_dof_velocities"][self.motion_index:]
        self.dof_target_pos_list = data["real_dof_positions_cmd"][self.motion_index:]
        self.dof_torque_list = data["real_dof_torques"][self.motion_index:]
        
        # Total number of motions
        self.motion_num = len(self.dof_positions_list)
        # Maximum time length of motions
        max_len_timestep = max(len(x) for x in self.dof_positions_list)
        
        # List of lengths for each motion
        self.motion_len = []
        for i in range(self.motion_num):
            self.motion_len.append(self.dof_positions_list[i].shape[0])
        self.motion_len = torch.tensor(self.motion_len, dtype=torch.long, device=self.device)
            
        # Expand all motions to the same length and form a large tensor
        self.dof_positions = torch.zeros((self.motion_num, max_len_timestep, self.num_dofs), dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.zeros((self.motion_num, max_len_timestep, self.num_dofs), dtype=torch.float32, device=self.device)
        self.dof_target_pos = torch.zeros((self.motion_num, max_len_timestep, self.num_dofs), dtype=torch.float32, device=self.device)
        self.dof_torque = torch.zeros((self.motion_num, max_len_timestep, self.num_dofs), dtype=torch.float32, device=self.device)
        
        for i in range(self.motion_num):
            pos = torch.as_tensor(self.dof_positions_list[i], dtype=torch.float32, device=self.device)
            vel = torch.as_tensor(self.dof_velocities_list[i], dtype=torch.float32, device=self.device)
            tgt = torch.as_tensor(self.dof_target_pos_list[i], dtype=torch.float32, device=self.device)
            tq = torch.as_tensor(self.dof_torque_list[i], dtype=torch.float32, device=self.device)
            
            cur_len = pos.shape[0]

            self.dof_positions[i, :cur_len, :] = pos
            self.dof_velocities[i, :cur_len, :] = vel
            self.dof_target_pos[i, :cur_len, :] = tgt
            self.dof_torque[i, :cur_len, :] = tq        



        self.joint_sequence = data["joint_sequence"]

        # Index of joint name in joint_sequence corresponding to self._dof_names, shape:(10,)
        self.joint_sequence_index = torch.tensor([self._dof_names.index(joint) for joint in self.joint_sequence], dtype=torch.long, device=self.device)
        
        # Add payload loading
        if "payloads" in data:
            self.payload_sequence = torch.from_numpy(data["payloads"]).float().to(self.device).unsqueeze(1)
        else:
            print("================== No payload data found in the motion file ==================")
            self.payload_sequence = torch.zeros((self.motion_num, 1), dtype=torch.float32, device=self.device)


    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)


    def sample_indices(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random motion and time indices uniformly.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            A tuple containing:
            - Random motion indices, between 0 and the total number of motions.
            - Random time indices, between 0 and the total number of frames per motion.
        """
        if self.mode == "train":
            motion_indices = torch.randint(low=0, high=self.motion_num, size=(num_samples,), device=self.device, dtype=torch.long)
            # time_indices = torch.randint(low=0, high=int(self.motion_len / 2), size=(num_samples,), device=self.device, dtype=torch.long)
            time_indices = torch.zeros((num_samples,), device=self.device, dtype=torch.long)
        if self.mode == "play":
            # motion_indices = torch.randint(low=0, high=self.motion_num, size=(num_samples,), device=self.device, dtype=torch.long)
            motion_indices = torch.arange(end=num_samples, device=self.device, dtype=torch.long) + self.sample_time
            motion_indices = torch.clamp(motion_indices, max=self.motion_num - 1)

            time_indices = torch.zeros((num_samples,), device=self.device, dtype=torch.long)
            self.sample_time += num_samples

            if self.sample_time >= self.motion_num:
                self.sample_time = 0
            
            
        # shape: [(num_samples, ), (num_samples, )], dtype: torch.long
        return motion_indices, time_indices


    def sample(
        self, num_samples: int, motion_indices: torch.Tensor | None = None, time_indices: torch.Tensor | None  = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data by motion and time index.

        Args:
            num_samples: Number of frame samples to generate. Ignored if ``indices`` is provided.
            indices: Specific motion and time indices to sample. If not provided, random indices will be generated.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            and DOF target positions (with shape (N, num_dofs)).
        """
        if motion_indices is None or time_indices is None:
            motion_indices, time_indices = self.sample_indices(num_samples)

        return (
            self.dof_positions[motion_indices][time_indices],
            self.dof_velocities[motion_indices][time_indices],
            self.dof_target_pos[motion_indices][time_indices],
            motion_indices,
            time_indices
        )


    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))

    npz_filename = './motion_perjoint_all/edited/motor_edited_extend_left_shoulder_pitch_joint_0kg.npz'
    npz_file_path = os.path.join(script_dir, npz_filename)

    motion = MotionLoaderMotor(npz_file_path, "cpu", "train", "h1_2_without_hand")

    print(motion.sample_indices(10))
    print(motion.sample(10))
    print(motion.dof_target_pos)
    # print("- number of DOFs:", motion.num_dofs)
    # print("- number of bodies:", motion.num_bodies)
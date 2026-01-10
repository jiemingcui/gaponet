# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR handtracking controller for SE(3) control."""
import contextlib
import numpy as np
from collections.abc import Callable
from scipy.spatial.transform import Rotation, Slerp
from typing import Final

from typing import Dict
from omni.kit.viewport.utility import get_active_viewport

from ..device_base import DeviceBase
from .openxr_device import OpenXRDevice

with contextlib.suppress(ModuleNotFoundError):
    from isaacsim.xr.openxr import OpenXR, OpenXRSpec
    from omni.kit.xr.core import XRCore

    from . import teleop_command

''' 

// Provided by XR_EXT_hand_tracking
typedef enum XrHandJointEXT {
    XR_HAND_JOINT_PALM_EXT = 0,
    XR_HAND_JOINT_WRIST_EXT = 1,
    XR_HAND_JOINT_THUMB_METACARPAL_EXT = 2,
    XR_HAND_JOINT_THUMB_PROXIMAL_EXT = 3,
    XR_HAND_JOINT_THUMB_DISTAL_EXT = 4,
    XR_HAND_JOINT_THUMB_TIP_EXT = 5,
    XR_HAND_JOINT_INDEX_METACARPAL_EXT = 6,
    XR_HAND_JOINT_INDEX_PROXIMAL_EXT = 7,
    XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT = 8,
    XR_HAND_JOINT_INDEX_DISTAL_EXT = 9,
    XR_HAND_JOINT_INDEX_TIP_EXT = 10,
    XR_HAND_JOINT_MIDDLE_METACARPAL_EXT = 11,
    XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT = 12,
    XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT = 13,
    XR_HAND_JOINT_MIDDLE_DISTAL_EXT = 14,
    XR_HAND_JOINT_MIDDLE_TIP_EXT = 15,
    XR_HAND_JOINT_RING_METACARPAL_EXT = 16,
    XR_HAND_JOINT_RING_PROXIMAL_EXT = 17,
    XR_HAND_JOINT_RING_INTERMEDIATE_EXT = 18,
    XR_HAND_JOINT_RING_DISTAL_EXT = 19,
    XR_HAND_JOINT_RING_TIP_EXT = 20,
    XR_HAND_JOINT_LITTLE_METACARPAL_EXT = 21,
    XR_HAND_JOINT_LITTLE_PROXIMAL_EXT = 22,
    XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT = 23,
    XR_HAND_JOINT_LITTLE_DISTAL_EXT = 24,
    XR_HAND_JOINT_LITTLE_TIP_EXT = 25,
    XR_HAND_JOINT_MAX_ENUM_EXT = 0x7FFFFFFF
} XrHandJointEXT;

'''


class Se3OpenXrTracking(OpenXRDevice):
    """OpenXR handtracking controller for SE(3) control."""
    GRIP_HYSTERESIS_METERS: Final[float] = 0.05  # 2.0 inch

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previous_grip_distance = 0.0
        self._max_grip_distance = 0.1
        self._min_grip_distance = 0.01
        self._previous_gripper_command = np.array([0., 0.], dtype=np.float32)

    def _calculate_gripper_command(self, hand_joints, previous_gripper_command):
        index_tip = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_INDEX_TIP_EXT]
        thumb_tip = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_THUMB_TIP_EXT]

        if not self._tracking:
            return previous_gripper_command

        index_tip_pos = index_tip[:3]
        thumb_tip_pos = thumb_tip[:3]
        distance = np.linalg.norm(index_tip_pos - thumb_tip_pos)
        # if distance > self._previous_grip_distance + self.GRIP_HYSTERESIS_METERS:
        #     self._previous_grip_distance = distance
        #     gripper_command = False
        # elif distance < self._previous_grip_distance - self.GRIP_HYSTERESIS_METERS:
        #     self._previous_grip_distance = distance
        #     gripper_command = True
        gripper_command = 1. - np.clip((distance - self._min_grip_distance) / self._max_grip_distance, 0., 1.)

        return gripper_command
    
    def _calculate_palm_pose(self, hand_joints: np.ndarray) -> np.ndarray:
        palm = hand_joints[OpenXRSpec.HandJointEXT.XR_HAND_JOINT_PALM_EXT]
        return palm.copy()
    
    def _calculate_hand_pose(self, hand_joints: np.ndarray) -> np.ndarray:
        return hand_joints[1:].copy()

    def advance(self) -> Dict[str, np.ndarray | bool]:
        """Advance the handtracking controller.

        Returns:
            Dict containing:
                head: Head tracking data
                left_palm: Left palm pose (position and orientation)
                right_palm: Right palm pose (position and orientation) 
                left_hand: Left hand joint poses (25 joints x 7 values)
                right_hand: Right hand joint poses (25 joints x 7 values)
                gripper_command: Gripper commands for both hands

            Orientation is in quaternion format (w, x, y, z).
        """
        raw_data = super()._get_raw_data()

        left_hand_joints = raw_data[OpenXRDevice.TrackingTarget.HAND_LEFT]
        left_gripper_command = self._calculate_gripper_command(left_hand_joints, self._previous_gripper_command[0])
        self._previous_gripper_command[0] = left_gripper_command

        right_hand_joints = raw_data[OpenXRDevice.TrackingTarget.HAND_RIGHT]
        right_gripper_command = self._calculate_gripper_command(right_hand_joints, self._previous_gripper_command[1])
        self._previous_gripper_command[1] = right_gripper_command

        return {
            "head": raw_data[OpenXRDevice.TrackingTarget.HEAD],
            "left_palm": self._calculate_palm_pose(left_hand_joints),
            "right_palm": self._calculate_palm_pose(right_hand_joints),
            "left_hand": self._calculate_hand_pose(left_hand_joints),
            "right_hand": self._calculate_hand_pose(right_hand_joints),
            "gripper_command": self._previous_gripper_command
        }
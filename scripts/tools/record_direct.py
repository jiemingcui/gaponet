# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=50,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.teleop_device.lower() == "handtracking":
    vars(args_cli)["experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import time
import torch

import omni.log

from isaaclab.devices import Se3HandTracking, Se3Keyboard, Se3SpaceMouse, Se3OpenXrTracking
from isaaclab.envs import ViewerCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import ViewportCameraController

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import RecorderManager
from isaaclab.assets import Articulation

import numpy as np
from typing import Dict

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(teleop_data: Dict[str, np.ndarray | bool], env: DirectRLEnv) -> Dict[str, torch.Tensor]:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    tensor_data = {}
    for key, value in teleop_data.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.float32 or value.dtype == np.float64:
                tensor_data[key] = torch.tensor(value, dtype=torch.float32, device=env.device)
            elif value.dtype == bool:
                tensor_data[key] = torch.tensor(value, dtype=torch.bool, device=env.device)
            else:
                raise ValueError(f"Unsupported dtype: {value.dtype}")
        elif isinstance(value, dict):
            tensor_data[key] = pre_process_actions(value, env)
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
    return tensor_data


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.teleop_device.lower() == "handtracking":
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    assert isinstance(env_cfg, DirectRLEnvCfg), "Only DirectRLEnv is supported for record_direct.py, if you want to record from ManagerBasedRLEnv, please use record_demos.py"

    env_cfg.env_name = args_cli.task # type: ignore

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped # type: ignore
    assert hasattr(env, "commands_to_action"), "Environment must have a commands_to_action function"

    # type hint
    env: DirectRLEnv

    recorder_manager_cfg = ActionStateRecorderManagerCfg()
    recorder_manager_cfg.dataset_export_dir_path = output_dir
    recorder_manager_cfg.dataset_filename = output_file_name

    recorder_manager = RecorderManager(recorder_manager_cfg, env) # type: ignore

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env, "_get_success"):
        assert callable(env._get_success) # type: ignore
        success_term = env._get_success # type: ignore
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    ready_term = None
    if hasattr(env, "_get_ready"):
        assert callable(env._get_ready) # type: ignore
        ready_term = env._get_ready # type: ignore
    else:
        omni.log.warn(
            "No ready termination term was found in the environment."
            " Will not be able to mark recorded demos as ready."
        )

    assert hasattr(env, "get_state"), "Environment must have a get_state function to record initial states"

    # add teleoperation key for reset current recording instance
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        raise NotImplementedError("Keyboard teleoperation is not supported for record_direct.py")
        teleop_interface = Se3Keyboard(pos_sensitivity=0.2, rot_sensitivity=0.5)
    elif args_cli.teleop_device.lower() == "spacemouse":
        raise NotImplementedError("SpaceMouse teleoperation is not supported for record_direct.py")
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.2, rot_sensitivity=0.5)
    elif args_cli.teleop_device.lower() == "handtracking":
        # from isaacsim.xr.openxr import OpenXRSpec

        # teleop_interface = Se3HandTracking(OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT, True)
        # teleop_interface.add_callback("RESET", reset_recording_instance)
        teleop_interface = Se3OpenXrTracking()
        viewer = ViewerCfg(eye=(2.2, -0.3, 3.5), lookat=(0.6, 0, 0), asset_name="viewer")
        ViewportCameraController(env, viewer)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'handtracking'."
        )

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("RESET", reset_recording_instance)
    print(teleop_interface)

    # reset before starting
    env.reset()
    teleop_interface.reset()

    # record initial state
    recorder_manager.add_to_episodes("initial_state", env.get_state(is_relative=True)) # type: ignore

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    success_step_count = 0
    ready_to_record = False
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            # get keyboard command
            teleop_data = teleop_interface.advance()
            # compute actions based on environment
            tensor_data = pre_process_actions(teleop_data, env) # type: ignore
            actions = env.commands_to_action(tensor_data) # type: ignore

            if ready_term is not None:
                if bool(ready_term):
                    ready_to_record = True
                    omni.log.info('Ready to record')
                else:
                    ready_to_record = False

            # record actions
            recorder_manager.add_to_episodes("actions", actions)

            # record observations
            obs = env._get_observations() # type: ignore
            recorder_manager.add_to_episodes("obs", obs)

            # perform action on environment
            _, _, terminated, truncated, _ = env.step(actions)
            early_terminated = terminated.item() or truncated.item()

            # record states
            recorder_manager.add_to_episodes("states", env.scene.get_state(is_relative=True))

            if early_terminated:
                # recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[False]], dtype=torch.bool, device=env.device)
                )
                should_reset_recording_instance = True

            elif success_term is not None:
                if bool(success_term()):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        # recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        recorder_manager.export_episodes([0])
                        should_reset_recording_instance = True
                else:
                    success_step_count = 0

            if should_reset_recording_instance:
                recorder_manager.reset()
                env.reset()

                # record initial state
                recorder_manager.add_to_episodes("initial_state", env.get_state(is_relative=True)) # type: ignore

                should_reset_recording_instance = False
                success_step_count = 0

            # print out the current demo count if it has changed
            if recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = recorder_manager.exported_successful_episode_count
                print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

            if args_cli.num_demos > 0 and recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.sim.is_stopped(): # type: ignore
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--letter", type=str, default=None, help="The first letter represents joint.")
parser.add_argument("--unwrapped", type=bool, default=False, help="Whether to unwrap the environment.")
parser.add_argument("--export_policy", type=bool, default=False, help="Whether to export the model.")
parser.add_argument("--no_delta_action", type=bool, default=False, help="Whether to not use delta action.")
parser.add_argument("--model", type=str, default=None, help="Direct path to the model checkpoint file to load.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from sim2real.rsl_rl.runners import *
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import sim2real.tasks  # noqa: F401

from process_env import process_motor_delta_action

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # load model checkpoint path
    if args_cli.model is not None and args_cli.model.strip() != "":
        # use direct model path if provided
        resume_path = os.path.abspath(args_cli.model)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Model file not found: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        print(f"[INFO] Loading model directly from: {resume_path}")
    else:
        # use original logic to find checkpoint from logs
        if agent_cfg.experiment_name is None:
            raise ValueError("Either --model must be provided, or experiment_name must be set in the config.")
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        if not os.path.exists(log_root_path):
            raise FileNotFoundError(
                f"Log directory not found: {log_root_path}. "
                f"Please provide a valid --model path or ensure the logs directory exists."
            )
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)

    try:
        env_cfg.mode = "play"
    except:
        pass    

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)   # type: ignore

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)   # type: ignore

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # create runner from rsl-rl
    runner_class = eval(agent_cfg.class_name)
    ppo_runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)   # type: ignore
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

     # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    if hasattr(policy_nn, "full_forward"):
        policy = policy_nn.full_forward

    # export policy to onnx/jit
    if args_cli.export_policy and hasattr(policy_nn, "actor"): 
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")

    if args_cli.unwrapped:
        env = env.unwrapped
        obs = env._get_observations()
    else:
        # reset environment
        obs, _ = env.get_observations()
    timestep = 0
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # import time
            # time_start = time.time()
            actions = policy(obs)
            # time_end = time.time()
            # print(f"Time taken: {time_end - time_start} seconds")
            if args_cli.no_delta_action:
                actions *= 0

            if not args_cli.unwrapped:
                obs, reward, done, extra = env.step(actions)   
            else:
                obs, _, _, _, _ = env.step(actions)   
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

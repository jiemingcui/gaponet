"""Evaluate GAPOnet compensation performance by comparing Sim, Sim+Delta, and Real trajectories."""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
from pathlib import Path
import importlib.util
import types

import gymnasium as gym
import matplotlib
import numpy as np
import torch

from isaaclab.app import AppLauncher

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add rsl_rl script directory to path to import cli_args
rsl_rl_script_dir = Path(__file__).resolve().parent / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.append(str(rsl_rl_script_dir))

try:
    import cli_args
except ImportError:
    print(f"Warning: Could not import cli_args from {rsl_rl_script_dir}. Some arguments might be missing.")
    cli_args = None


def _sync_to_motion(env_unwrapped, motion_indices: torch.Tensor, time_indices: torch.Tensor) -> None:
    motion_pos = env_unwrapped._motion_loader.dof_positions[motion_indices, time_indices]
    motion_vel = torch.zeros_like(motion_pos)
    env_unwrapped.robot.write_joint_state_to_sim(
        motion_pos,
        motion_vel,
    )
    env_unwrapped.robot.set_joint_position_target(
        motion_pos,
    )
    env_unwrapped._raw_step_simulator()


def _update_sensor_data_from_sim(env_unwrapped) -> None:
    joint_indices = getattr(env_unwrapped._motion_loader, "joint_sequence_index", None)
    if joint_indices is None:
        joint_pos = env_unwrapped.robot.data.joint_pos
        joint_vel = env_unwrapped.robot.data.joint_vel
    else:
        joint_pos = env_unwrapped.robot.data.joint_pos[:, joint_indices]
        joint_vel = env_unwrapped.robot.data.joint_vel[:, joint_indices]
    sensor = torch.cat([joint_pos, joint_vel * env_unwrapped.step_dt], dim=1)
    sensor = sensor.view(env_unwrapped.num_envs, env_unwrapped.num_sensor_positions, env_unwrapped.cfg.sensor_dim)
    env_unwrapped.set_sensor_data(sensor)


def _load_runner_checkpoint(ppo_runner, resume_path: str, load_optimizer: bool = False) -> None:
    checkpoint = None
    try:
        checkpoint = torch.load(resume_path, map_location="cpu")
    except Exception:
        checkpoint = None

    def _load_policy_weights() -> bool:
        if not isinstance(checkpoint, dict):
            return False
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if state is None:
            return False
        policy = None
        if hasattr(ppo_runner.alg, "policy"):
            policy = ppo_runner.alg.policy
        elif hasattr(ppo_runner.alg, "actor_critic"):
            policy = ppo_runner.alg.actor_critic
        if policy is None:
            raise RuntimeError("Unsupported runner: no policy/actor_critic to load.")

        load_result = policy.load_state_dict(state, strict=False)
        missing = getattr(load_result, "missing_keys", None)
        unexpected = getattr(load_result, "unexpected_keys", None)
        if missing is not None or unexpected is not None:
            if missing or unexpected:
                print(f"[WARN] Pretrain checkpoint load mismatch. missing={missing}, unexpected={unexpected}")

        obs_norm_state = checkpoint.get("obs_norm_state_dict")
        if obs_norm_state is not None and getattr(ppo_runner, "obs_normalizer", None) is not None:
            ppo_runner.obs_normalizer.load_state_dict(obs_norm_state, strict=False)

        priv_state = checkpoint.get("privileged_obs_norm_state_dict")
        if priv_state is not None:
            for attr in ("critic_obs_normalizer", "privileged_obs_normalizer"):
                normalizer = getattr(ppo_runner, attr, None)
                if normalizer is not None:
                    normalizer.load_state_dict(priv_state, strict=False)
                    break

        ppo_runner.current_learning_iteration = int(checkpoint.get("iteration", checkpoint.get("iter", 0)))
        print("[INFO] Loaded checkpoint weights (runner state skipped).")
        return True

    if _load_policy_weights():
        return

    if isinstance(checkpoint, dict) and "iter" in checkpoint:
        try:
            ppo_runner.load(resume_path, load_optimizer=load_optimizer)
        except TypeError:
            ppo_runner.load(resume_path)
        except KeyError:
            if _load_policy_weights():
                return
            raise
        return

    try:
        ppo_runner.load(resume_path, load_optimizer=load_optimizer)
    except TypeError:
        ppo_runner.load(resume_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate GAPOnet compensation performance.")

    # Task and Environment args
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Humanoid-Operator-Delta-Action-Fourior",
        help="Gym task id to run.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs to create.")

    # Motion selection args
    parser.add_argument("--motion-index", type=int, default=0, help="Fixed motion index to use.")
    parser.add_argument("--time-index", type=int, default=0, help="Fixed start time index to use.")
    parser.add_argument("--motion-file", type=str, default=None, help="Override motion file path.")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of steps to simulate.")
    parser.add_argument("--full-episode", action="store_true", help="Override num-steps to the full episode length.")

    # Simulation args
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument(
        "--hang-timeout",
        type=int,
        default=600,
        help="Seconds before dumping traceback for long-running steps; 0 disables.",
    )

    # Checkpoint args
    parser.add_argument(
        "--use_pretrained_checkpoint",
        action="store_true",
        help="Use the pre-trained checkpoint from Nucleus.",
    )

    # When cli_args is not available (for example, if the rsl_rl utilities are not
    # found on PYTHONPATH), we still want to be able to pass an explicit checkpoint.
    # In that case, expose a lightweight --checkpoint flag here that is compatible
    # with the usage later in this script.
    if cli_args is None:
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Path to a model checkpoint (.pt) to evaluate when RSL-RL cli_args is unavailable.",
        )

    # Plotting args
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="logs/train_result",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--interactive-html",
        action="store_true",
        help="Also save interactive HTML plots (click legend to highlight/hide lines).",
    )
    parser.add_argument(
        "--new-history-update",
        action="store_true",
        help="Use new model_history timing (update after step; default is legacy timing).",
    )
    parser.add_argument(
        "--model-based-sensor",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help="Override sensor source for compensation phase: auto(from agent cfg), true(use sensor model), false(use raw simulator sensor).",
    )

    # Add RSL-RL args
    if cli_args:
        cli_args.add_rsl_rl_args(parser)

    # Add AppLauncher args
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    # Configure hang traceback timeout (faulthandler)
    if args.hang_timeout <= 0:
        faulthandler.cancel_dump_traceback_later()
    else:
        faulthandler.dump_traceback_later(args.hang_timeout, repeat=True)

    # Launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import Isaac Sim dependent modules (MUST be after AppLauncher)
    import rsl_rl.runners.on_policy_runner as on_policy_runner
    from rsl_rl.runners import OnPolicyRunner
    from sim2real.rsl_rl.runners import OperatorRunner, OperatorVanillaRunner
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    # -------------------------------------------------------------------------
    # Import tasks and env config (similar to check_motion_alignment logic)
    # NOTE: For Fourier humanoid operator evaluation we use the humanoid_operator
    # task package instead of humanoid_agibot.
    # Repository layout (from this file):
    #   <repo_root>/evaluate_compensation.py
    #   <repo_root>/source/...
    # so we take parent() as repo root.
    # -------------------------------------------------------------------------
    source_root = Path(__file__).resolve().parent / "source"
    sys.path.append(str(source_root))

    tasks_init = (
        source_root
        / "sim2real"
        / "sim2real"
        / "tasks"
        / "humanoid_operator"
        / "__init__.py"
    )
    if not tasks_init.is_file():
        raise RuntimeError(f"Task module not found: {tasks_init}")

    if "sim2real" not in sys.modules:
        sim2real_pkg = types.ModuleType("sim2real")
        sim2real_pkg.__path__ = [str(source_root / "sim2real")]
        sys.modules["sim2real"] = sim2real_pkg
    elif not hasattr(sys.modules["sim2real"], "__path__"):
        sys.modules["sim2real"].__path__ = [str(source_root / "sim2real")]
    tasks_pkg_path = source_root / "sim2real" / "sim2real" / "tasks"
    if "sim2real.tasks" not in sys.modules:
        tasks_pkg = types.ModuleType("sim2real.tasks")
        tasks_pkg.__path__ = [str(tasks_pkg_path)]
        sys.modules["sim2real.tasks"] = tasks_pkg
    elif not hasattr(sys.modules["sim2real.tasks"], "__path__"):
        sys.modules["sim2real.tasks"].__path__ = [str(tasks_pkg_path)]

    spec = importlib.util.spec_from_file_location("sim2real.tasks.humanoid_operator", tasks_init)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load task module: {tasks_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim2real.tasks.humanoid_operator"] = module
    spec.loader.exec_module(module)

    # Register DeepONetActorCritic after task modules are ready
    from sim2real.rsl_rl.modules import DeepONetActorCritic

    on_policy_runner.DeepONetActorCritic = DeepONetActorCritic

    # Load Env Cfg (Fourier variant for humanoid operator)
    env_cfg_path = (
        source_root
        / "sim2real"
        / "sim2real"
        / "tasks"
        / "humanoid_operator"
        / "humanoid_operator_env_cfg_fourior.py"
    )
    if not env_cfg_path.is_file():
        raise RuntimeError(f"Env cfg module not found: {env_cfg_path}")

    env_cfg_mod_name = "sim2real.tasks.humanoid_operator.humanoid_operator_env_cfg_fourior"
    env_cfg_spec = importlib.util.spec_from_file_location(env_cfg_mod_name, env_cfg_path)
    if env_cfg_spec is None or env_cfg_spec.loader is None:
        raise RuntimeError(f"Failed to load env cfg module: {env_cfg_path}")
    env_cfg_module = importlib.util.module_from_spec(env_cfg_spec)
    sys.modules[env_cfg_mod_name] = env_cfg_module
    env_cfg_spec.loader.exec_module(env_cfg_module)

    # -------------------------------------------------------------------------
    # Setup Environment Configuration
    # -------------------------------------------------------------------------
    env_cfg = env_cfg_module.HumanoidOperatorEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    if args.motion_file:
        env_cfg.train_motion_file = args.motion_file
        env_cfg.test_motion_file = args.motion_file
        print(f"[INFO] Overriding motion file with: {args.motion_file}")

    # -------------------------------------------------------------------------
    # Create environment (reuse for baseline + compensation)
    # -------------------------------------------------------------------------
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env_unwrapped = env.unwrapped
    device = env_unwrapped.device
    num_actions = env_unwrapped.cfg.action_space

    motion_len = int(env_unwrapped._motion_loader.motion_len[args.motion_index].item())
    if args.full_episode:
        num_steps = motion_len
    else:
        num_steps = min(int(args.num_steps), motion_len)

    print(f"[INFO] Environment created. Device: {device}, Num Envs: {args.num_envs}")
    print(f"[INFO] Motion Index: {args.motion_index}, Length: {motion_len}, Steps: {num_steps}")

    motion_indices = torch.full((args.num_envs,), args.motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((args.num_envs,), args.time_index, dtype=torch.long, device=device)

    _sync_to_motion(env_unwrapped, motion_indices, time_indices)

    zero_action = torch.zeros((args.num_envs, num_actions), device=device)
    sim_baseline_traj = []
    real_traj = []

    for _ in range(num_steps):
        time_indices_step = time_indices.clone()
        _, _, dones, _ = env_unwrapped.step_operator(
            zero_action, motion_coords=(motion_indices, time_indices_step)
        )

        sim_pos = env_unwrapped.robot.data.joint_pos[
            :, env_unwrapped._motion_loader.joint_sequence_index
        ].detach().cpu().numpy()
        real_pos = env_unwrapped._motion_loader.dof_positions[
            motion_indices, time_indices_step
        ].detach().cpu().numpy()

        sim_baseline_traj.append(sim_pos[0])
        real_traj.append(real_pos[0])

        motion_indices = env_unwrapped.motion_indices.clone()
        time_indices = env_unwrapped.time_indices.clone()

        if bool(torch.any(dones)):
            break

    sim_baseline_arr = np.stack(sim_baseline_traj, axis=0)
    real_arr = np.stack(real_traj, axis=0)

    print(f"[INFO] Baseline simulation complete. Steps: {len(sim_baseline_traj)}")

    # -------------------------------------------------------------------------
    # Phase 2: Run Compensated Simulation (With GAPOnet)
    # -------------------------------------------------------------------------
    if cli_args:
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    else:
        agent_cfg = RslRlOnPolicyRunnerCfg(experiment_name="default", run_name="default")

    if args.model_based_sensor != "auto":
        agent_cfg.model_based_sensor = args.model_based_sensor == "true"
        print(f"[INFO] Override model_based_sensor={agent_cfg.model_based_sensor}")

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args.task)
    elif args.checkpoint:
        resume_path = retrieve_file_path(args.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    env_comp_unwrapped = env_unwrapped
    env_wrapped = RslRlVecEnvWrapper(env)

    _RUNNER_REGISTRY = {
        "OnPolicyRunner": OnPolicyRunner,
        "OperatorRunner": OperatorRunner,
        "OperatorVanillaRunner": OperatorVanillaRunner,
    }
    runner_class_name = agent_cfg.to_dict().get("class_name", "OnPolicyRunner")
    runner_class = _RUNNER_REGISTRY.get(runner_class_name, OnPolicyRunner)
    print(f"[INFO] Using runner class: {runner_class.__name__}")
    ppo_runner = runner_class(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    _load_runner_checkpoint(ppo_runner, resume_path, load_optimizer=False)
    policy = ppo_runner.get_inference_policy(device=device)
    use_model_sensor = bool(getattr(agent_cfg, "model_based_sensor", False))

    motion_indices = torch.full((args.num_envs,), args.motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((args.num_envs,), args.time_index, dtype=torch.long, device=device)

    _sync_to_motion(env_comp_unwrapped, motion_indices, time_indices)
    env_comp_unwrapped.last_delta_action[:] = 0
    if hasattr(env_comp_unwrapped, "model_history"):
        env_comp_unwrapped.model_history[:] = 0

    sim_comp_traj = []
    target_traj = []
    target_delta_traj = []
    prev_joint_pos = env_comp_unwrapped.robot.data.joint_pos[
        :, env_comp_unwrapped._motion_loader.joint_sequence_index
    ].clone()
    prev_joint_vel = env_comp_unwrapped.robot.data.joint_vel[
        :, env_comp_unwrapped._motion_loader.joint_sequence_index
    ].clone()

    for _ in range(num_steps):
        time_indices_step = time_indices.clone()
        if use_model_sensor:
            with torch.no_grad():
                # For humanoid operator environments, compute_model_observation only
                # takes `add_noise` and internally manages any history buffers.
                model_obs = env_comp_unwrapped.compute_model_observation(add_noise=False).to(device)
                sensor_data = ppo_runner.alg.policy.model_sensor(model_obs).reshape(
                    env_comp_unwrapped.num_envs, env_comp_unwrapped.num_sensor_positions, -1
                )
            env_comp_unwrapped.set_sensor_data(sensor_data)
        else:
            joint_pos = env_comp_unwrapped.robot.data.joint_pos[
                :, env_comp_unwrapped._motion_loader.joint_sequence_index
            ]
            joint_vel = env_comp_unwrapped.robot.data.joint_vel[
                :, env_comp_unwrapped._motion_loader.joint_sequence_index
            ]
            sensor = torch.cat([joint_pos, joint_vel * env_comp_unwrapped.step_dt], dim=1)
            if env_comp_unwrapped.cfg.delta_sensor_value:
                prev = torch.cat([prev_joint_pos, prev_joint_vel * env_comp_unwrapped.step_dt], dim=1)
                sensor = sensor - prev
            sensor = sensor.view(
                env_comp_unwrapped.num_envs, env_comp_unwrapped.num_sensor_positions, env_comp_unwrapped.cfg.sensor_dim
            )
            env_comp_unwrapped.set_sensor_data(sensor)

        with torch.no_grad():
            obs_dict = env_comp_unwrapped.compute_operator_observation()
            obs = torch.cat([obs_dict["branch"], obs_dict["trunk"]], dim=1)
            actions = policy(obs)

        dof_target_pos = env_comp_unwrapped._motion_loader.dof_target_pos[motion_indices, time_indices_step]
        delta_action = torch.zeros((args.num_envs, env_comp_unwrapped.num_dofs), device=device)
        delta_action[:, env_comp_unwrapped._motion_loader.joint_sequence_index] = actions
        apply_action = dof_target_pos + delta_action

        target_pos = dof_target_pos.detach().cpu().numpy()
        target_delta_pos = apply_action.detach().cpu().numpy()

        _, _, dones, _ = env_comp_unwrapped.step_operator(
            actions, motion_coords=(motion_indices, time_indices_step)
        )

        motion_indices = env_comp_unwrapped.motion_indices.clone()
        time_indices = env_comp_unwrapped.time_indices.clone()

        sim_pos = env_comp_unwrapped.robot.data.joint_pos[
            :, env_comp_unwrapped._motion_loader.joint_sequence_index
        ].detach().cpu().numpy()
        sim_comp_traj.append(sim_pos[0])
        target_traj.append(target_pos[0])
        target_delta_traj.append(target_delta_pos[0])

        prev_joint_pos = env_comp_unwrapped.robot.data.joint_pos[
            :, env_comp_unwrapped._motion_loader.joint_sequence_index
        ].clone()
        prev_joint_vel = env_comp_unwrapped.robot.data.joint_vel[
            :, env_comp_unwrapped._motion_loader.joint_sequence_index
        ].clone()

        if bool(torch.any(dones)):
            break

    sim_comp_arr = np.stack(sim_comp_traj, axis=0)
    target_arr = np.stack(target_traj, axis=0)
    target_delta_arr = np.stack(target_delta_traj, axis=0)
    print(f"[INFO] Compensated simulation complete. Steps: {len(sim_comp_traj)}")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    min_len = min(sim_baseline_arr.shape[0], sim_comp_arr.shape[0], real_arr.shape[0], target_arr.shape[0])
    sim_baseline_arr = sim_baseline_arr[:min_len]
    sim_comp_arr = sim_comp_arr[:min_len]
    real_arr = real_arr[:min_len]
    target_arr = target_arr[:min_len]
    target_delta_arr = target_delta_arr[:min_len]

    steps = np.arange(min_len)

    sim_base_deg = np.degrees(sim_baseline_arr)
    sim_comp_deg = np.degrees(sim_comp_arr)
    real_deg = np.degrees(real_arr)
    target_deg = np.degrees(target_arr)
    target_delta_deg = np.degrees(target_delta_arr)

    dof_names = list(env_comp_unwrapped._motion_loader.dof_names)
    joint_count = len(dof_names)

    plot_dir = Path(args.plot_dir) / f"episode{args.motion_index:03d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = plot_dir.resolve()

    print(f"[INFO] Saving plots to: {plot_dir}")

    if min_len == 0:
        raise RuntimeError("No trajectory data collected; cannot generate plots.")

    if args.interactive_html:
        try:
            import plotly.graph_objects as go
        except Exception as exc:
            raise RuntimeError("plotly is required for --interactive-html. Please install plotly.") from exc

    for idx in range(joint_count):
        name = dof_names[idx]
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        ax0.plot(steps, real_deg[:, idx], "r-", label="real", linewidth=1.5, alpha=0.8)
        ax0.plot(steps, sim_base_deg[:, idx], "b--", label="sim", linewidth=1.5, alpha=0.8)
        ax0.plot(steps, sim_comp_deg[:, idx], "g-", label="sim+delta", linewidth=1.5, alpha=0.8)
        ax0.set_title(f"Joint: {name}")
        ax0.set_ylabel("position (deg)")
        ax0.legend()
        ax0.grid(True, alpha=0.3)

        err_base = np.abs(sim_base_deg[:, idx] - real_deg[:, idx])
        err_comp = np.abs(sim_comp_deg[:, idx] - real_deg[:, idx])
        ax1.plot(steps, err_base, "b--", label=f"|sim-real| mean: {np.mean(err_base):.3f}")
        ax1.plot(steps, err_comp, "g-", label=f"|sim+delta-real| mean: {np.mean(err_comp):.3f}")
        ax1.set_ylabel("|error| (deg)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, target_deg[:, idx], color="orange", linestyle=":", label="target", linewidth=1.5, alpha=0.8)
        ax2.plot(steps, target_delta_deg[:, idx], color="purple", linestyle="-.", label="target+delta", linewidth=1.5, alpha=0.8)
        ax2.set_xlabel("step")
        ax2.set_ylabel("target (deg)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}_traj_comp.png")
        plt.close(fig)

        if args.interactive_html:
            from plotly.subplots import make_subplots

            html_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
            html_fig.add_trace(go.Scatter(x=steps, y=real_deg[:, idx], mode="lines", name="real"), row=1, col=1)
            html_fig.add_trace(go.Scatter(x=steps, y=sim_base_deg[:, idx], mode="lines", name="sim"), row=1, col=1)
            html_fig.add_trace(go.Scatter(x=steps, y=sim_comp_deg[:, idx], mode="lines", name="sim+delta"), row=1, col=1)

            html_fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=err_base,
                    mode="lines",
                    name=f"|sim-real| mean: {np.mean(err_base):.3f}",
                    line=dict(dash="dash"),
                ),
                row=2,
                col=1,
            )
            html_fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=err_comp,
                    mode="lines",
                    name=f"|sim+delta-real| mean: {np.mean(err_comp):.3f}",
                ),
                row=2,
                col=1,
            )

            html_fig.add_trace(go.Scatter(x=steps, y=target_deg[:, idx], mode="lines", name="target"), row=3, col=1)
            html_fig.add_trace(
                go.Scatter(x=steps, y=target_delta_deg[:, idx], mode="lines", name="target+delta"),
                row=3,
                col=1,
            )

            html_fig.update_layout(
                title=f"Joint: {name}",
                hovermode="x unified",
                legend_title_text="Click to hide/highlight",
                height=900,
            )
            html_fig.update_yaxes(title_text="position (deg)", row=1, col=1)
            html_fig.update_yaxes(title_text="|error| (deg)", row=2, col=1)
            html_fig.update_yaxes(title_text="target (deg)", row=3, col=1)
            html_fig.update_xaxes(title_text="step", row=3, col=1)
            html_fig.write_html(plot_dir / f"{name}_traj_comp.html", include_plotlyjs="cdn")

    print("[INFO] Done.")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

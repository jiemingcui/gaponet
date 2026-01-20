#!/usr/bin/env python3
"""Lightweight inference and evaluation script for JIT model deployment.

This script performs inference on test data using a JIT-compiled model and computes
evaluation metrics grouped by payload mass. No Isaac Sim is required for inference.

The script:
1. Loads a JIT-compiled model (exported using inference_jit.py)
2. Processes test data from .npz files
3. Performs inference on all motions
4. Computes evaluation metrics:
   - Large Gap Ratio: Ratio of joint position errors >= 0.5 rad
   - Gap IQR: Interquartile range of joint position errors
   - Gap Range: Range (max - min) of joint position errors
5. Displays results grouped by payload mass

Usage:
    # Run inference and evaluation
    python scripts/rsl_rl/deploy.py \\
        --model ./model/policy.pt \\
        --test_data ./source/sim2real/sim2real/tasks/humanoid_operator/motions/motion_amass/edited_27dof/test.npz \\
        --device cuda:0
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


def load_jit_model(jit_path: str, device: str = "cuda:0"):
    """Load a JIT model from file.
    
    Args:
        jit_path: Path to the JIT model file (.pt)
        device: Device to load model on (default: cuda:0)
        
    Returns:
        model: Loaded JIT model in eval mode
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        RuntimeError: If model loading fails
    """
    if not os.path.exists(jit_path):
        raise FileNotFoundError(f"JIT model file not found: {jit_path}")
    
    print(f"[INFO] Loading JIT model from: {jit_path}")
    try:
        model = torch.jit.load(jit_path, map_location=device)
        model.eval()
        print(f"[INFO] JIT model loaded successfully on {device}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load JIT model: {e}") from e


def inference(model, obs: Dict, device: str = "cuda:0") -> torch.Tensor:
    """Run inference on the model.
    
    Args:
        model: JIT model
        obs: Dictionary with keys:
            - "model": model_obs tensor [1, model_input_dim]
            - "operator": dict with "branch" [1, branch_dim] and "trunk" [1, trunk_dim]
        device: Device to run on
        
    Returns:
        actions: Predicted actions [1, action_dim]
    """
    model_obs = obs["model"].to(device)
    branch_obs = obs["operator"]["branch"].to(device)
    trunk_obs = obs["operator"]["trunk"].to(device)
    
    with torch.no_grad():
        actions = model(model_obs, branch_obs, trunk_obs)
    
    return actions


def load_obs_from_npz(npz_path: str, motion_idx: int = 0, time_idx: int = 0, device: str = "cuda:0", 
                      joint_sequence_index: Optional[List[int]] = None, sensor_dim: int = 20, 
                      num_sensor_positions: int = 20, add_model_history: bool = True, 
                      model_history_length: int = 4, model_history_dim: int = 30,
                      step_dt: float = 0.01, predicted_joint_pos: Optional[np.ndarray] = None, 
                      predicted_joint_vel: Optional[np.ndarray] = None, verbose: bool = False):
    """Load observations from npz file and construct model/operator observations.
    
    Args:
        npz_path: Path to npz file
        motion_idx: Motion index
        time_idx: Time step index
        device: Device to run on
        joint_sequence_index: Joint sequence indices
        sensor_dim: Sensor dimension
        num_sensor_positions: Number of sensor positions
        add_model_history: Whether to add model history
        model_history_length: Model history length
        model_history_dim: Model history dimension
        step_dt: Time step
        predicted_joint_pos: Predicted joint positions from previous steps [num_joints] (for model history)
        predicted_joint_vel: Predicted joint velocities from previous steps [num_joints] (for model history)
        verbose: Whether to print debug info
        
    Returns:
        obs_dict: Dictionary with "model", "operator" keys
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Get motion data
    dof_positions = data["real_dof_positions"][motion_idx]  # [T, num_dofs]
    dof_velocities = data["real_dof_velocities"][motion_idx]  # [T, num_dofs]
    dof_target_pos = data["real_dof_positions_cmd"][motion_idx]  # [T, num_dofs]
    
    num_dofs = dof_positions.shape[1]
    if joint_sequence_index is None:
        joint_sequence_index = list(range(num_dofs))
    num_joints = len(joint_sequence_index)
    
    if time_idx >= len(dof_positions):
        raise ValueError(f"time_idx {time_idx} exceeds motion length {len(dof_positions)}")
    
    # Use predicted joint positions if available (for model history), otherwise use real data
    if predicted_joint_pos is not None:
        joint_pos = torch.from_numpy(predicted_joint_pos).float().to(device)  # [num_joints]
    else:
        joint_pos = torch.from_numpy(dof_positions[time_idx][joint_sequence_index]).float().to(device)  # [num_joints]
    
    if predicted_joint_vel is not None:
        joint_vel = torch.from_numpy(predicted_joint_vel).float().to(device)  # [num_joints]
    else:
        joint_vel = torch.from_numpy(dof_velocities[time_idx][joint_sequence_index]).float().to(device)  # [num_joints]
    
    joint_target = torch.from_numpy(dof_target_pos[time_idx][joint_sequence_index]).float().to(device)  # [num_joints]
    current_action = joint_target.unsqueeze(0)  # [1, num_joints]
    
    # Build model_obs
    if add_model_history:
        # Build model_history: [history_length, model_history_dim]
        history_start = max(0, time_idx - model_history_length + 1)
        history_data = []
        for t in range(history_start, time_idx + 1):
            if t == time_idx and predicted_joint_pos is not None:
                # Use predicted values for current step
                pos = torch.from_numpy(predicted_joint_pos).float().to(device)
                vel = torch.from_numpy(predicted_joint_vel).float().to(device)
            else:
                pos = torch.from_numpy(dof_positions[t][joint_sequence_index]).float().to(device)
                vel = torch.from_numpy(dof_velocities[t][joint_sequence_index]).float().to(device)
            tgt = torch.from_numpy(dof_target_pos[t][joint_sequence_index]).float().to(device)
            hist_step = torch.cat([pos, vel, tgt])[:model_history_dim]
            history_data.append(hist_step)
        
        while len(history_data) < model_history_length:
            pad_tensor = history_data[0] if history_data else torch.zeros(model_history_dim, device=device)
            history_data.insert(0, pad_tensor)
        
        model_history = torch.stack(history_data[-model_history_length:])  # [history_length, model_history_dim]
        joint_pos_for_model = joint_pos[:10] if num_joints > 10 else joint_pos
        if num_joints < 10:
            padding = torch.zeros(10 - num_joints, device=device)
            joint_pos_for_model = torch.cat([joint_pos_for_model, padding])
        model_obs = torch.cat([joint_pos_for_model.unsqueeze(0), model_history.flatten().unsqueeze(0)], dim=1)  # [1, 10 + 120] = [1, 130]
    else:
        payload_mass = torch.tensor([0.001], device=device).unsqueeze(0)
        model_obs = torch.cat([joint_pos.unsqueeze(0), joint_vel.unsqueeze(0), payload_mass], dim=1)  # [1, num_joints*2 + 1]
    
    # Build branch_obs (sensor_data)
    sensor_data_raw = torch.cat([
        joint_pos.unsqueeze(0),  # [1, num_joints]
        joint_vel.unsqueeze(0) * step_dt  # [1, num_joints]
    ], dim=1)  # [1, num_joints * 2]
    
    # Pad/truncate sensor_data_raw to match total_sensor_size (400)
    total_sensor_size = num_sensor_positions * sensor_dim  # 20 * 20 = 400
    if sensor_data_raw.shape[1] < total_sensor_size:
        padding = torch.zeros(1, total_sensor_size - sensor_data_raw.shape[1], device=device)
        sensor_data_raw = torch.cat([sensor_data_raw, padding], dim=1)
    elif sensor_data_raw.shape[1] > total_sensor_size:
        sensor_data_raw = sensor_data_raw[:, :total_sensor_size]
    
    branch_obs = sensor_data_raw.flatten(1)  # [1, 400]
    
    # Build trunk_obs: [current_action]
    trunk_obs = current_action[:, :10] if num_joints >= 10 else current_action  # [1, 10]
    if num_joints < 10:
        padding = torch.zeros(1, 10 - num_joints, device=device)
        trunk_obs = torch.cat([trunk_obs, padding], dim=1)
    
    obs_dict = {
        "model": model_obs,
        "operator": {
            "branch": branch_obs,
            "trunk": trunk_obs
        }
    }
    
    if verbose:
        print(f"[INFO] Loaded observations for motion {motion_idx}, time {time_idx}")
        print(f"[INFO] Model obs shape: {model_obs.shape}, Branch obs shape: {branch_obs.shape}, Trunk obs shape: {trunk_obs.shape}")
    
    return obs_dict


def evaluate_on_test_data(model, npz_path: str, device: str = "cuda:0", 
                          joint_sequence_index: Optional[List[int]] = None, sensor_dim: int = 20, 
                          num_sensor_positions: int = 20, add_model_history: bool = True,
                          model_history_length: int = 4, model_history_dim: int = 30, 
                          step_dt: float = 0.01) -> Dict:
    """Evaluate model on all motions in test.npz file and compute gap metrics by payload mass.
    
    Args:
        model: JIT model for inference
        npz_path: Path to test.npz file
        device: Device to run on
        joint_sequence_index: Joint sequence indices
        Other args: Same as load_obs_from_npz
        
    Returns:
        metrics: Dictionary containing evaluation metrics grouped by payload mass
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Get all motion data
    dof_positions = data["real_dof_positions"]  # List of [T, num_dofs]
    dof_velocities = data["real_dof_velocities"]  # List of [T, num_dofs]
    dof_target_pos = data["real_dof_positions_cmd"]  # List of [T, num_dofs]
    
    num_motions = len(dof_positions)
    num_dofs = dof_positions[0].shape[1] if num_motions > 0 else 0
    
    if joint_sequence_index is None:
        joint_sequence_index = list(range(num_dofs))
    num_joints = len(joint_sequence_index)
    
    print(f"[INFO] Evaluating on {num_motions} motions...")
    
    # Get payload sequence
    if "payloads" in data and data["payloads"] is not None:
        payload_sequence = data["payloads"]  # List of payloads per motion
    else:
        payload_sequence = [0.001] * num_motions
    
    # Collect all payload masses
    mass_levels = sorted(set([float(p[0]) if isinstance(p, np.ndarray) and len(p) > 0 else float(p) for p in payload_sequence]))
    
    # Initialize bins for each metric by mass level
    def init_bins():
        return {m: [] for m in mass_levels}
    
    large_gap_ratio_bins = init_bins()
    gap_iqr_bins = init_bins()
    gap_range_bins = init_bins()
    
    # Helper functions for computing metrics (same as in humanoid_operator_env.py)
    # These functions follow the exact logic from humanoid_operator_env.py
    def run_large_gap_ratio(seq, key):
        """Compute ratio of gaps >= 0.5 rad.
        
        Args:
            seq: List of dictionaries, each containing joint position differences
            key: Key to access joint position differences in each step
            
        Returns:
            Ratio of errors >= 0.5 rad
        """
        errors = np.abs(np.stack([step[key] for step in seq]))
        large_gap_count = np.sum(errors >= 0.5)
        total_points = len(errors.flatten())
        return float(large_gap_count / total_points) if total_points > 0 else 0.0
    
    def run_gap_iqr(seq, key):
        """Compute Interquartile Range.
        
        Args:
            seq: List of dictionaries, each containing joint position differences
            key: Key to access joint position differences in each step
            
        Returns:
            IQR (75th percentile - 25th percentile)
        """
        errors = np.abs(np.stack([step[key] for step in seq]))
        q75 = np.percentile(errors, 75)
        q25 = np.percentile(errors, 25)
        return float(q75 - q25)
    
    def run_gap_range(seq, key):
        """Compute range (max - min).
        
        Args:
            seq: List of dictionaries, each containing joint position differences
            key: Key to access joint position differences in each step
            
        Returns:
            Range (max - min)
        """
        errors = np.abs(np.stack([step[key] for step in seq]))
        return float(np.max(errors) - np.min(errors))
    
    # Calculate total timesteps for progress bar
    total_timesteps = sum(len(dof_positions[i]) for i in range(num_motions))
    motion_lengths = [len(dof_positions[i]) for i in range(num_motions)]
    
    print(f"[INFO] Total motions: {num_motions}")
    print(f"[INFO] Total timesteps: {total_timesteps} (sum of all motion lengths)")
    print(f"[INFO] Motion lengths - min: {min(motion_lengths)}, max: {max(motion_lengths)}, mean: {np.mean(motion_lengths):.1f}")
    
    # Process each motion with progress bar
    pbar = tqdm(total=total_timesteps, desc="Inference progress", unit="timestep")
    
    for motion_idx in range(num_motions):
        motion_length = len(dof_positions[motion_idx])
        
        # Get payload mass for this motion
        payload = payload_sequence[motion_idx]
        if isinstance(payload, np.ndarray):
            payload = payload[0] if len(payload) > 0 else 0.001
        payload_mass = float(payload)
        
        # Initialize predicted joint positions (start from real initial position)
        predicted_joint_pos = dof_positions[motion_idx][0][joint_sequence_index].copy()  # [num_joints]
        predicted_joint_vel = np.zeros(num_joints)  # [num_joints]
        
        # Store joint position differences for this motion
        motion_joint_pos_diffs = []  # List of [num_joints] arrays
        
        # Process each time step in the motion
        for time_idx in range(motion_length):
            try:
                # Load observations for this time step
                # Use predicted positions for model_history (to simulate accumulation)
                # Use real positions for branch_obs (as input, like in real environment)
                obs = load_obs_from_npz(
                    npz_path, motion_idx=motion_idx, time_idx=time_idx, device=device,
                    joint_sequence_index=joint_sequence_index, sensor_dim=sensor_dim,
                    num_sensor_positions=num_sensor_positions, add_model_history=add_model_history,
                    model_history_length=model_history_length, model_history_dim=model_history_dim,
                    step_dt=step_dt, predicted_joint_pos=predicted_joint_pos, predicted_joint_vel=predicted_joint_vel
                )
                
                # Run inference - model outputs delta_action
                predicted_delta_actions = inference(model, obs, device)
                predicted_delta_actions = predicted_delta_actions.cpu().numpy()[0]  # [action_dim]
                
                # Get current target action (absolute position at current time)
                current_target = dof_target_pos[motion_idx][time_idx][joint_sequence_index[:10]]  # [10]
                if len(current_target) < 10:
                    current_target = np.pad(current_target, (0, 10 - len(current_target)), 'constant')
                
                # Apply delta_action to get predicted final position
                # Model outputs delta_action, so predicted final = current_target + predicted_delta
                predicted_final_pos = current_target + predicted_delta_actions
                
                # Update predicted joint positions for next step (use predicted final position)
                # This simulates the robot's joint position after applying the action
                predicted_joint_pos[:10] = predicted_final_pos
                # For remaining joints, keep them as is (or use real data)
                if num_joints > 10:
                    predicted_joint_pos[10:] = dof_positions[motion_idx][time_idx][joint_sequence_index[10:]]
                
                # Calculate joint position difference: robot_joint_pos - real_joint_pos (without abs, abs is taken in metric functions)
                # In environment: joint_pos_diff = robot.data.joint_pos - real_joint_pos (then abs is taken in metric functions)
                # We use predicted_final_pos as robot_joint_pos (after applying action)
                # Use joint_sequence_index (first 10 joints, excluding wrist if needed)
                real_joint_pos = dof_positions[motion_idx][time_idx][joint_sequence_index[:10]]  # [10]
                if len(real_joint_pos) < 10:
                    real_joint_pos = np.pad(real_joint_pos, (0, 10 - len(real_joint_pos)), 'constant')
                
                # joint_pos_diff = robot_joint_pos - real_joint_pos (in radians, abs will be taken in metric functions)
                # Note: Following the environment logic, we store the raw difference (not abs)
                # The abs is taken inside the metric functions (run_large_gap_ratio, etc.)
                joint_pos_diff = predicted_final_pos - real_joint_pos  # [10], in radians
                motion_joint_pos_diffs.append({
                    'joint_pos_diff': joint_pos_diff  # Store as dict to match environment format
                })
                
                # Update progress bar
                pbar.update(1)
                
            except Exception as e:
                print(f"\n[WARNING] Failed at motion {motion_idx}, time {time_idx}: {e}")
                pbar.update(1)
                continue
        
        # Compute metrics for this motion sequence
        if len(motion_joint_pos_diffs) > 0:
            # motion_joint_pos_diffs is now a list of dicts, matching the environment format
            # Compute metrics using the same logic as environment
            large_gap_ratio = run_large_gap_ratio(motion_joint_pos_diffs, 'joint_pos_diff')
            gap_iqr = run_gap_iqr(motion_joint_pos_diffs, 'joint_pos_diff')
            gap_range = run_gap_range(motion_joint_pos_diffs, 'joint_pos_diff')
            
            # Add to bins by payload mass
            large_gap_ratio_bins[payload_mass].append(large_gap_ratio)
            gap_iqr_bins[payload_mass].append(gap_iqr)
            gap_range_bins[payload_mass].append(gap_range)
    
    pbar.close()
    
    # Print results tables
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
    
    def append_section_csv(section_title, bins, csv_path=None):
        """Append a section into CSV file."""
        if csv_path is None:
            return
        # This would write to CSV file if needed
        # For now, we just print the tables
        pass
    
    print("######################################################################")
    print_table("Large Gap Ratio (>=0.5 rad) by Mass", large_gap_ratio_bins)
    append_section_csv("Large Gap Ratio (>=0.5 rad) by Mass", large_gap_ratio_bins)
    print_table("Gap IQR (rad) by Mass", gap_iqr_bins)
    append_section_csv("Gap IQR (rad) by Mass", gap_iqr_bins)
    print_table("Gap Range (rad) by Mass", gap_range_bins)
    append_section_csv("Gap Range (rad) by Mass", gap_range_bins)
    print("######################################################################")
    
    # Return metrics for potential further processing
    metrics = {
        "large_gap_ratio_bins": large_gap_ratio_bins,
        "gap_iqr_bins": gap_iqr_bins,
        "gap_range_bins": gap_range_bins,
        "mass_levels": mass_levels,
    }
    
    return metrics


def main():
    """Main function for lightweight inference and evaluation.
    
    This script performs inference on test data using a JIT-compiled model
    and computes evaluation metrics (Large Gap Ratio, Gap IQR, Gap Range)
    grouped by payload mass. No Isaac Sim is required.
    """
    parser = argparse.ArgumentParser(
        description="Lightweight inference for JIT model deployment and evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on test data
  python scripts/rsl_rl/deploy.py \\
      --model ./model/policy.pt \\
      --test_data ./source/sim2real/sim2real/tasks/humanoid_operator/motions/motion_amass/edited_27dof/test.npz \\
      --device cuda:0
        """
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Path to JIT model file (.pt)")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test.npz file containing motion data")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run on (default: cuda:0)")
    parser.add_argument("--sensor_dim", type=int, default=20,
                       help="Sensor dimension per position (default: 20)")
    parser.add_argument("--num_sensor_positions", type=int, default=20,
                       help="Number of sensor positions (default: 20)")
    parser.add_argument("--model_history_length", type=int, default=4,
                       help="Model history length (default: 4)")
    parser.add_argument("--model_history_dim", type=int, default=30,
                       help="Model history dimension (default: 30)")
    parser.add_argument("--step_dt", type=float, default=0.01,
                       help="Time step in seconds (default: 0.01)")
    
    args = parser.parse_args()
    
    # Validate inputs
    import os
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data file not found: {args.test_data}")
    
    try:
        # Load model
        model = load_jit_model(args.model, args.device)
        
        # Evaluate on test data
        metrics = evaluate_on_test_data(
            model, args.test_data, device=args.device,
            sensor_dim=args.sensor_dim, num_sensor_positions=args.num_sensor_positions,
            model_history_length=args.model_history_length, model_history_dim=args.model_history_dim,
            step_dt=args.step_dt
        )
        
        print("[INFO] Evaluation completed successfully.")
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

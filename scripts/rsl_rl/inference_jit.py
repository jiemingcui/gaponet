"""Export checkpoint model to JIT/TorchScript format for lightweight inference.

This script exports a trained checkpoint model to JIT format, which can then be used
for inference without requiring Isaac Sim. The exported model includes:
- DeepONet branch and trunk networks
- Sensor model (model network)
- Observation normalizer

Note: The export process itself requires Isaac Sim to be initialized, but the exported
JIT model can be used for inference on any machine with PyTorch.

Usage:
    # Export checkpoint to JIT format
    python scripts/rsl_rl/inference_jit.py \\
        --export \\
        --checkpoint ./model/model_17950.pt \\
        --task Isaac-Humanoid-Operator-Delta-Action \\
        --output ./model/policy.pt \\
        --device cuda:0 \\
        --num_envs 20
"""

import argparse
import os
import torch


def load_jit_model(jit_path: str, device: str = "cuda:0"):
    """Load a JIT-compiled model for inference.
    
    Args:
        jit_path: Path to the .pt JIT model file
        device: Device to load model on
        
    Returns:
        model: Loaded JIT model in eval mode
    """
    print(f"[INFO] Loading JIT model from: {jit_path}")
    if not os.path.exists(jit_path):
        raise FileNotFoundError(f"JIT model file not found: {jit_path}")
    
    model = torch.jit.load(jit_path, map_location=device)
    model.eval()
    print(f"[INFO] JIT model loaded successfully on {device}")
    return model


def inference_jit(model, obs, device: str = "cuda:0"):
    """Run inference using JIT model.
    
    Args:
        model: JIT-compiled model (expects 3 inputs: model_obs, branch_obs, trunk_obs)
        obs: Observation - can be:
            - dict with 'model', 'operator' keys (full format)
            - dict with 'branch' and 'trunk' keys (simplified, model_obs will be zeros)
            - tuple of (model_obs, branch_obs, trunk_obs) or (branch_obs, trunk_obs)
        device: Device to run on
        
    Returns:
        actions: Predicted actions, shape (batch_size, action_dim)
    """
    model.eval()
    
    with torch.no_grad():
        if isinstance(obs, dict):
            if "model" in obs and "operator" in obs:
                model_obs = obs["model"].to(device)
                operator_obs = obs["operator"]
                branch = operator_obs["branch"].to(device)
                trunk = operator_obs["trunk"].to(device)
                actions = model(model_obs, branch, trunk)
            elif "branch" in obs and "trunk" in obs:
                branch = obs["branch"].to(device)
                trunk = obs["trunk"].to(device)
                model_obs = torch.zeros(branch.shape[0], branch.shape[1], device=device)
                actions = model(model_obs, branch, trunk)
            else:
                raise ValueError("For DeepONet, obs dict must contain ('model', 'operator') or ('branch', 'trunk') keys")
        elif isinstance(obs, tuple):
            if len(obs) == 3:
                model_obs, branch, trunk = [x.to(device) for x in obs]
                actions = model(model_obs, branch, trunk)
            elif len(obs) == 2:
                branch, trunk = [x.to(device) for x in obs]
                model_obs = torch.zeros(branch.shape[0], branch.shape[1], device=device)
                actions = model(model_obs, branch, trunk)
            else:
                raise ValueError("Tuple must have 2 or 3 elements: (branch, trunk) or (model_obs, branch, trunk)")
        else:
            raise ValueError("obs must be a dict or tuple, not a single tensor")
    
    return actions


def export_model_to_jit(checkpoint_path: str, output_path: str, task_name: str = None, device: str = "cuda:0", num_envs: int = None):
    """Export a checkpoint model to JIT format.
    
    This function requires Isaac Sim to be initialized first.
    Use this once to export, then use load_jit_model() for lightweight inference.
    
    Args:
        checkpoint_path: Path to the checkpoint .pt file
        output_path: Path to save the JIT model
        task_name: Task name (optional, for config)
        device: Device to use
        num_envs: Number of environments to simulate (default: None, uses config default)
    """
    print(f"[INFO] Exporting model from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Import necessary modules (requires Isaac Sim)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from sim2real.rsl_rl.modules import DeepONetActorCritic
    from sim2real.rsl_rl.modules import EmpiricalNormalization
    
    if not task_name:
        raise ValueError("--task is required to load configuration.")
    
    print("[INFO] Loading configuration from registry...")
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    policy_cfg = agent_cfg.policy
    print("[INFO] Configuration loaded successfully.")
    
    # Create environment to get action_dim and model_input_dim (required for policy creation)
    print("[INFO] Creating environment to get required dimensions...")
    import gymnasium as gym
    import sim2real.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    
    try:
        env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs, use_fabric=False)
        print("[INFO] Initializing gym environment (this may take a moment)...")
        env = gym.make(task_name, cfg=env_cfg)
        print("[INFO] Environment created.")
        
        # Get unwrapped environment to access cfg and methods
        env_unwrapped = env.unwrapped
        
        # Get action_dim from environment (required for policy creation)
        action_dim = env_unwrapped.cfg.action_space
        
        # Get model_input_dim from environment and update policy_cfg
        if hasattr(env_unwrapped, 'compute_model_observation'):
            model_obs_dim = env_unwrapped.compute_model_observation().shape[1]
            if policy_cfg.model_input_dim != model_obs_dim:
                print(f"[INFO] Updating model_input_dim from {policy_cfg.model_input_dim} to {model_obs_dim}")
            policy_cfg.model_input_dim = model_obs_dim
        
        # Get observation dimensions for normalizer
        if hasattr(env_unwrapped, 'compute_operator_observation'):
            obs_dict = env_unwrapped.compute_operator_observation()
        else:
            obs_dict = env_unwrapped._get_observations()
        
        if isinstance(obs_dict, dict):
            num_obs = sum(v.shape[1] for k, v in obs_dict.items() if k != "critic")
            num_privileged_obs = obs_dict.get("critic", torch.zeros(1, 1)).shape[1]
        else:
            num_obs = obs_dict.shape[1]
            num_privileged_obs = num_obs
        
        print(f"[INFO] Dimensions: action_dim={action_dim}, num_obs={num_obs}, num_privileged_obs={num_privileged_obs}")
        env.close()
        print("[INFO] Environment closed.")
    except Exception as e:
        print(f"[ERROR] Could not create environment: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError("Failed to create environment. Cannot proceed without action_dim and model_input_dim.")

    # Create policy model (same way as operator_runner.py: 0, 0 for num_obs and num_privileged_obs)
    print(f"[INFO] Creating policy model with action_dim={action_dim}...")
    policy_cfg_dict = policy_cfg.to_dict() if hasattr(policy_cfg, 'to_dict') else dict(policy_cfg)
    policy_cfg_copy = dict(policy_cfg_dict)
    policy_class = eval(policy_cfg_copy.pop("class_name"))
    policy = policy_class(0, 0, action_dim, **policy_cfg_copy).to(device)
    
    # Load weights
    policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Load normalizer
    obs_normalizer = None
    if "obs_norm_state_dict" in checkpoint:
        obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(device)
        try:
            obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
        except RuntimeError:
            print("[WARNING] Could not load normalizer, will export without it.")
            obs_normalizer = None
    
    # Export to JIT
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_deeponet_to_jit(policy, obs_normalizer, output_path)
    print(f"[INFO] Model exported to JIT: {output_path}")


def export_deeponet_to_jit(policy, normalizer, output_path: str):
    """Export DeepONetActorCritic to JIT format.
    
    This follows the full_forward logic:
    1. model_obs -> sensor_model -> sensors
    2. sensors replace the first part of branch_input
    3. branch_input + trunk_input -> DeepONet -> actions
    
    Args:
        policy: DeepONetActorCritic model
        normalizer: Observation normalizer (optional, for branch+trunk)
        output_path: Path to save JIT model
    """
    import copy
    
    class DeepONetJITExporter(torch.nn.Module):
        """JIT exporter for DeepONetActorCritic following full_forward logic."""
        
        def __init__(self, policy, normalizer=None):
            super().__init__()
            # Copy all necessary networks
            self.branch_net = copy.deepcopy(policy.branch_net)
            self.trunk_net = copy.deepcopy(policy.trunk_net)
            self.sensor_model = copy.deepcopy(policy.model)  # sensor model
            self.action_dim = policy.action_dim
            self.branch_input_dims = policy.branch_input_dims
            self.total_branch_dim = policy.total_branch_dim
            self.total_trunk_dim = policy.total_trunk_dim
            
            # Copy input normalizer if exists (for branch+trunk normalization)
            if normalizer:
                self.normalizer = copy.deepcopy(normalizer)
            else:
                self.normalizer = torch.nn.Identity()
        
        def forward(self, model_obs: torch.Tensor, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
            """Forward pass matching full_forward logic."""
            # Run sensor model
            sensors = self.sensor_model(model_obs)
            
            # Replace first part of branch_input with sensors
            branch_input = branch_input.clone()
            branch_input[:, :sensors.shape[1]] = sensors
            
            # Apply normalizer
            combined = torch.cat([branch_input, trunk_input], dim=1)
            combined = self.normalizer(combined)
            branch_input = combined[:, :self.total_branch_dim]
            trunk_input = combined[:, self.total_branch_dim:]
            
            # Chunk branch input according to branch_input_dims
            branch_inputs = []
            start_idx = 0
            for dim in self.branch_input_dims:
                branch_inputs.append(branch_input[:, start_idx:start_idx+dim])
                start_idx += dim
            
            # Forward through DeepONet
            batch_size = branch_inputs[0].shape[0]
            branch_out = self.branch_net(branch_inputs)
            trunk_out = self.trunk_net(trunk_input)
            combined_out = branch_out * trunk_out
            
            # Get actions
            actions = combined_out.view(batch_size, -1, self.action_dim).sum(dim=1)
            return actions
    
    # Create exporter
    exporter = DeepONetJITExporter(policy, normalizer)
    exporter.eval()
    
    # Create dummy inputs for tracing
    model_input_dim = policy.model_input_dim
    branch_dim = policy.total_branch_dim
    trunk_dim = policy.total_trunk_dim
    
    dummy_model_obs = torch.randn(1, model_input_dim)
    dummy_branch = torch.randn(1, branch_dim)
    dummy_trunk = torch.randn(1, trunk_dim)
    
    # Trace the model
    exporter.to("cpu")
    with torch.no_grad():
        traced = torch.jit.trace(exporter, (dummy_model_obs, dummy_branch, dummy_trunk))
    
    # Save
    traced.save(output_path)
    print(f"[INFO] JIT model exported with sensor model. Input: (model_obs, branch_obs, trunk_obs)")


def main():
    """Main function for exporting checkpoint to JIT format.
    
    This script exports a trained checkpoint model to JIT/TorchScript format
    for lightweight inference without Isaac Sim. Requires Isaac Sim to be
    initialized for the export process.
    """
    parser = argparse.ArgumentParser(
        description="Export checkpoint model to JIT format for lightweight inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export a checkpoint to JIT format
  python scripts/rsl_rl/inference_jit.py --export \\
      --checkpoint ./model/model_17950.pt \\
      --task Isaac-Humanoid-Operator-Delta-Action \\
      --output ./model/policy.pt \\
      --device cuda:0 \\
      --num_envs 20
        """
    )
    parser.add_argument("--export", action="store_true", 
                       help="Export checkpoint to JIT (requires Isaac Sim)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint .pt file")
    parser.add_argument("--task", type=str, required=True,
                       help="Task name (e.g., Isaac-Humanoid-Operator-Delta-Action)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for exported JIT model (.pt)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run on (default: cuda:0)")
    parser.add_argument("--num_envs", type=int, default=None,
                       help="Number of environments to simulate (default: use config default)")
    
    args = parser.parse_args()
    
    if not args.export:
        raise ValueError("--export flag is required. This script only supports export mode.")
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Created output directory: {output_dir}")
    
    # Initialize Isaac Sim for export
    print("[INFO] Initializing Isaac Sim for export...")
    from isaaclab.app import AppLauncher
    export_args = argparse.Namespace(headless=True, device=args.device)
    app_launcher = AppLauncher(export_args)
    simulation_app = app_launcher.app
    
    try:
        export_model_to_jit(args.checkpoint, args.output, args.task, args.device, args.num_envs)
        print(f"[INFO] Export completed successfully: {args.output}")
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
        print("[INFO] Isaac Sim closed.")


if __name__ == "__main__":
    main()


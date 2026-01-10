# GapONet: Sim-to-Real Humanoid Robot Control

A reinforcement learning framework for training humanoid robot controllers using Isaac Lab, featuring DeepONet, Transformer, and MLP architectures for sim-to-real transfer.

This project works as a standalone Isaac Lab extension, similar to [humanoid_amp](https://github.com/linden713/humanoid_amp). No symbolic links or special setup required!

## Overview

GapONet implements a comprehensive training and evaluation framework for humanoid robot control with a focus on sim-to-real transfer. It supports multiple neural network architectures (DeepONet, Transformer, MLP) and provides environments for training and testing on various humanoid robot platforms.

### Key Features

- **Multiple Network Architectures**: DeepONet, Transformer, and MLP-based actor-critic networks
- **Sim-to-Real Transfer**: Environments designed for training in simulation and deployment on real robots
- **Multi-Robot Support**: Support for Unitree H1 and Fourior humanoid robots
- **Motion Data Integration**: Loads and processes motion data from AMASS dataset and custom motion files
- **Payload Adaptation**: Handles variable payload masses for robust control
- **Comprehensive Evaluation**: Metrics and visualization tools for performance analysis

## Prerequisites

- **Isaac Sim 4.5.0+** (installed separately)
- **Python 3.10+**
- **CUDA-capable GPU** with appropriate drivers
- **Isaac Lab** (see [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/) for installation)

Make sure you have Isaac Sim and Isaac Lab installed via pip before proceeding.

## Assets

Before installation, download the required assets:

1. **Robot Assets**: Download [sim2real_assets](https://drive.google.com/drive/folders/1Us5FTDRO_whoxDDO_Nqa8KmbWCFDNyVX?usp=sharing) and place the corresponding files in `gaponet/source/sim2real_assets/`.

2. **Test Data**: A [test data](https://drive.google.com/file/d/12h3iOTuttKmxflI-SP5UEpgINQq06mKp/view?usp=sharing) sample is provided. Please refer to this template for the format of test and training data.

3. **Test Model**: [test model]() (link to be added)

## Installation

Use the setup script to automatically create the conda environment and install all dependencies:

```bash
# Clone the repository
git clone git@github.com:jiemingcui/gaponet.git
cd gaponet

# Run the setup script (creates 'gapo' environment by default)
./setup.sh
```

The setup script will:
- Create a conda environment
- Install all dependencies
- Install the package in editable mode

Sync correct codebase into your IsaacLab setting:

```bash
# Set ISAACLAB_PATH environment variable first
export ISAACLAB_PATH=/path/to/isaaclab

# Sync the scripts
./sync_rsl_scripts.sh
```

**Note**: The sync script copies training scripts from Isaac Lab. If you prefer, you can use Isaac Lab's training scripts directly without syncing.

## Usage

### Training

#### Operator Environment (DeepONet)

Train with DeepONet architecture on operator environment:

**11-dimensional trunk (with payload):**
```bash
python scripts/rsl_rl/train.py --task Isaac-Humanoid-Operator-Delta-Action \
  --num_envs=4080 --max_iterations 100000 --experiment_name Sim2Real \
  --letter amass --run_name delta_action_mlp_payload --device cuda env.mode=train --headless
```

**10-dimensional trunk (without payload):**
```bash
python scripts/rsl_rl/train.py --task Isaac-Humanoid-Operator-Delta-Action \
  --num_envs=4080 --max_iterations 100000 --experiment_name Sim2Real \
  --letter amass --run_name delta_action_mlp --device cuda env.mode=train --headless
```

**Note**: If you haven't synced the scripts, you can use Isaac Lab's training scripts directly:
```bash
# Using Isaac Lab's isaaclab.sh
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Humanoid-Operator-Delta-Action \
  --num_envs=4080 --max_iterations 100000 \
  --experiment_name Sim2Real --letter amass \
  --run_name delta_action_mlp_payload --device cuda env.mode=train --headless
```

### Evaluation/Playback

Evaluate a trained model:

```bash
python scripts/rsl_rl/play.py --task Isaac-Humanoid-Operator-Delta-Action \
  --checkpoint your_model.pt --num_envs 20 --headless
```

Or using Isaac Lab's script directly:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Humanoid-Operator-Delta-Action \
  --checkpoint your_model.pt --num_envs 20 --headless
```


## Adding a New Robot

To add support for a new robot, follow these steps:

1. **Create a new task directory** in `source/sim2real/sim2real/tasks/`:
   - Create a new folder (e.g., `humanoid_your_robot/`)
   - Copy and modify files from `humanoid_operator/` or `humanoid_amass/` as reference
   - Implement your environment class (e.g., `your_robot_env.py`)
   - Create environment configuration (e.g., `your_robot_env_cfg.py`)

2. **Register the task** in `source/sim2real/sim2real/tasks/your_robot/__init__.py`:
   - Use `gym.register()` to register your environment
   - Reference existing registrations in `humanoid_operator/__init__.py`

3. **Create robot configuration** in `source/sim2real_assets/sim2real_assets/robots/`:
   - Create a Python file (e.g., `your_robot.py`)
   - Define robot configuration using `ArticulationCfg`
   - Add joint and body name dictionaries

4. **Add robot assets**:
   - Place URDF files in `source/sim2real_assets/sim2real_assets/urdfs/`
   - Place USD files in `source/sim2real_assets/sim2real_assets/usds/` (if using USD)
   - Create versions with and without payloads if needed

5. **Prepare motion data**:
   - Format your data according to the test data template
   - Save as `.npz` files with required keys (see Motion Data section)
   - Place in appropriate motion directory

6. **Configure agent settings**:
   - Create or modify agent config in `source/sim2real/sim2real/tasks/your_robot/agents/`
   - Choose appropriate method (DeepONet, Transformer, or MLP)
   - Set network parameters based on your robot's DOF count

7. **Start training**:
   - Use the registered task name in training commands
   - Adjust `num_envs` and other hyperparameters as needed

## Architecture

### DeepONet Actor-Critic

The DeepONet architecture uses a branch-trunk network structure:
- **Branch Network**: Processes sensor data at multiple resolutions
- **Trunk Network**: Processes action targets and payload information
- **Fusion**: Combines branch and trunk outputs for action prediction

### Transformer Actor-Critic

Transformer-based architecture with:
- Multi-head self-attention
- Position-wise feed-forward networks
- Separate actor and critic transformers

### MLP Actor-Critic

Standard multi-layer perceptron with:
- Configurable hidden dimensions
- History buffer for temporal information
- Action and value heads

## Environments

### HumanoidOperator

Operator environment for training with variable payloads and sensor configurations:
- Supports multiple sensor positions
- Handles wrist and hand payloads
- Computes equivalent torques using Pinocchio
- Sub-environment structure for efficient training

### HumanoidAmass

AMASS motion tracking environment:
- Loads motion data from AMASS dataset
- Tracks reference motions
- Supports history-based observations
- Computes tracking rewards

## Configuration

### Environment Configuration

Key parameters in environment configs:
- `mode`: "train" or "play"
- `num_envs`: Number of parallel environments
- `episode_length_s`: Episode length in seconds
- `max_payload_mass`: Maximum payload mass for training
- `num_sensor_positions`: Number of sensor configurations

### Network Configuration

Network-specific parameters:
- `branch_input_dims`: Input dimensions for branch networks
- `trunk_input_dim`: Input dimension for trunk network
- `hidden_dims`: Hidden layer dimensions
- `model_history_length`: Length of history buffer

## Motion Data

Motion data should be provided in NumPy `.npz` format with the following keys:
- `real_dof_positions`: Joint positions
- `real_dof_velocities`: Joint velocities
- `real_dof_positions_cmd`: Target joint positions
- `real_dof_torques`: Joint torques
- `joint_sequence`: List of joint names for delta actions
- `payloads`: Payload masses (optional)

## Evaluation Metrics

The framework computes several metrics during evaluation:
- **MPJAE**: Mean Per-Joint Angle Error (in degrees)
- **Large Gap Ratio**: Ratio of gaps >= 0.5 rad
- **Gap IQR**: Interquartile range of gaps
- **Gap Range**: Range of gaps
- **Upper Body Joint Area**: Area under error curve
- **EEF Error**: End-effector position error

Results are saved as CSV files and visualization plots.

## Acknowledgments

- Built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- Uses [rsl_rl](https://github.com/leggedrobotics/rsl_rl) for RL algorithms
- Motion data from [AMASS](https://amass.is.tue.mpg.de/)
- Inspired by [humanoid_amp](https://github.com/linden713/humanoid_amp)

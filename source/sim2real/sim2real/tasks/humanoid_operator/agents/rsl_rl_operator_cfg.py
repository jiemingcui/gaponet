# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from typing import Literal, List
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class DeepONetActorCriticCfg:
    class_name: str = "DeepONetActorCritic"

    """Configuration for DeepONet Actor-Critic network"""
    # Branch network configuration
    # Fourior 版本中，branch 输入来自 sensor_data.flatten(1, 2)，维度 = num_sensor_positions * sensor_dim = 20 * 62 = 1240
    branch_input_dims: List[int] = [20 * 62]  # = 1240
    branch_hidden_dim: int = 256  # Hidden dimension for branch networks
    
    # Trunk network配置
    # Fourior 版本中，trunk 输入为 current_action(31 DOF 参考关节目标) + payload(1) => 32
    trunk_input_dim: int = 32  # Input dimension for trunk network
    trunk_hidden_dims: List[int] = [128, 128, 128]  # Hidden dimensions for trunk network
    
    # Output configuration
    activation: str = "elu"  # Activation function
    
    # Critic network 配置
    # Critic 输入为：
    # sensor_data.flatten(1,2) [1240]
    # + current_action [31]
    # + robot joint_pos [31]
    # + robot joint_vel [31]
    # + robot joint_acc [31]
    # + real_joint_pos [31]
    # + real_joint_vel [31]
    # + wrist_payload_mass [1]
    # + hand_payload_mass [2]
    # + robot_mass [34]  # Fourior 机器人当前 num_bodies = 34
    # 合计 1240 + 6*31 + 1 + 2 + 34 = 1463
    critic_input_dim: int = 1463  # Input dimension for critic network
    critic_hidden_dims: List[int] = [256, 128, 128]  # Hidden dimensions for critic network

    # Model network configuration
    # 使用 31 DOF 的 history：每帧 93 维，history_length=4，则理论输入为 31 + 93*4 = 403。
    # 注意：实际运行时，OperatorRunner 会用 env.compute_model_observation() 的输出维度覆盖这个值。
    model_input_dim: int = 31 + 93*4  # 仅作为初始配置，真实值在运行时由 env 决定
    # 传感器目标是 sub_env_sensor_data.flatten(1, 2)，维度为 num_sensor_positions * sensor_dim = 20 * 62 = 1240
    model_output_dim: int = 20 * 62  # = 1240，与 env.compute_model_pairs() 中的 sensor.flatten(1, 2) 一致
    model_hidden_dims: List[int] = [128, 128]  # Hidden dimensions for model network

    # Model history configuration
    model_history_length: int = 4   # 必须与 env 配置中的 model_history_length 一致
    model_history_dim: int = 93     # 必须与 env 配置中的 model_history_dim 一致
    model_pretrained_path: str = ""

@configclass
class HumanoidOperatorRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OperatorRunner"

    """Configuration for DeepONet PPO runner"""
    num_steps_per_env = 32
    # Fourior 版本保持与原始配置一致：每个函数只采样一步（num_steps_function=1）
    num_steps_function = 1

    max_iterations = 120
    save_interval = 50
    replay_buffer_size = 40
    experiment_name = "humanoid_operator"
    empirical_normalization = True
    
    direct_sample_envs = True
    full_trajectory_sampling = True
    
    # DeepONet policy configuration
    policy = DeepONetActorCriticCfg() # type: ignore
    
    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
    
    # Logger configuration
    logger: Literal["tensorboard", "neptune", "wandb"] = "wandb"
    wandb_project: str = "humanoid-deeponet-direct"

    # model learning configuration
    model_based_sensor = True
    model_replay_buffer_size = 50
    model_learning_epochs = 300
    model_learning_steps = 1
    model_learning_interval = 1000
    model_sample_iterations = 10

    # zero-shot transformation configuration
    retrain_sensor_only = False

    # augmentation configuration
    randomize_dynamics = True

    # evaluation configuration
    eval_after_training = False

@configclass
class HumanoidOperatorVanillaRunnerCfg(HumanoidOperatorRunnerCfg):
    class_name = "OperatorVanillaRunner"

@configclass
class HumanoidOperatorFourierRunnerCfg(HumanoidOperatorRunnerCfg):
    """Configuration for Humanoid Operator with Fourier features

    Example of task-specific runner configuration. This inherits all settings from
    HumanoidOperatorRunnerCfg but uses DeepONetActorCriticFourierCfg for the policy.

    To use this configuration, register it in __init__.py:
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_operator_cfg:HumanoidOperatorFourierRunnerCfg"
    """
    class_name = "OperatorRunner"

    # DeepONet policy configuration with fourier
    policy = DeepONetActorCriticCfg() # type: ignore